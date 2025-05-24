import os
import logging
import threading
import uuid
from flask import (
    Flask,
    request,
    render_template,
    send_file,
    jsonify,
    url_for,
)
from werkzeug.utils import secure_filename
from datetime import datetime

# Import our enhanced modules
from Extract import PDFExtractor
from compare import PdfCompare
from generate import ReportGenerator

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pdf_compare.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
# NOTE: UPLOAD_FOLDER is no longer needed as files are processed in-memory.
app.config["REPORT_FOLDER"] = os.path.join(os.getcwd(), "reports")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64MB max upload size
app.config["ALLOWED_EXTENSIONS"] = {"pdf"}

# In-memory store for task tracking
processing_tasks = {}

# Create necessary directories
os.makedirs(app.config["REPORT_FOLDER"], exist_ok=True)


def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_files():
    """
    Handles file uploads by reading them into memory for processing
    without saving them to disk.
    """
    try:
        if "pdf1" not in request.files or "pdf2" not in request.files:
            return (
                jsonify({"status": "error", "message": "Both PDF files are required"}),
                400,
            )

        file1 = request.files["pdf1"]
        file2 = request.files["pdf2"]

        if file1.filename == "" or file2.filename == "":
            return (
                jsonify({"status": "error", "message": "File names cannot be empty"}),
                400,
            )

        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            return (
                jsonify({"status": "error", "message": "Only PDF files are allowed"}),
                400,
            )

        # Read file content into memory instead of saving to disk
        pdf1_content = file1.read()
        pdf2_content = file2.read()
        
        # Secure original filenames for display and logging
        orig_filename1 = secure_filename(file1.filename)
        orig_filename2 = secure_filename(file2.filename)

        logger.info(f"Files read into memory: {orig_filename1}, {orig_filename2}")

        task_id = str(uuid.uuid4())
        
        with app.app_context():
            report_url_template = url_for("get_report", filename="PLACEHOLDER")
            report_url_base = report_url_template.replace("PLACEHOLDER", "")

        # Store task info in memory. Note: PDF content is passed directly to the
        # processing thread to manage memory usage.
        processing_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "orig_filename1": orig_filename1,
            "orig_filename2": orig_filename2,
            "start_time": datetime.now().isoformat(),
            "report_url_base": report_url_base,
            "status_message": "Initializing comparison",
        }

        # Start background processing with in-memory content
        threading.Thread(
            target=process_comparison_with_progress,
            args=(task_id, pdf1_content, pdf2_content, orig_filename1, orig_filename2, app),
        ).start()

        return jsonify(
            {
                "status": "processing",
                "message": "Comparison started",
                "task_id": task_id,
            }
        )

    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}", exc_info=True)
        return (
            jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}),
            500,
        )


def process_comparison_with_progress(
    task_id, pdf1_content, pdf2_content, filename1, filename2, app_instance
):
    """
    Process PDF comparison with progress tracking using in-memory file content.
    """
    try:
        logger.info(
            f"Starting comparison task {task_id} between {filename1} and {filename2}"
        )

        def extraction_progress(progress):
            scaled_progress = 5 + (progress * 20)
            update_task_progress(
                task_id,
                scaled_progress,
                f"Extracting content from first PDF ({int(progress*100)}%)",
            )
        
        extractor = PDFExtractor(
            similarity_threshold=0.85,
            header_match_threshold=0.9,
            nested_table_threshold=0.85,
            nested_area_ratio=0.75,
            max_workers=os.cpu_count(),
        )

        update_task_progress(task_id, 5, "Extracting content from first PDF")
        logger.info(f"Extracting content from {filename1}")
        # PDF content is already in memory
        pdf1_data = extractor.extract_pdf_content(pdf1_content, extraction_progress)

        def extraction2_progress(progress):
            scaled_progress = 25 + (progress * 20)
            update_task_progress(
                task_id,
                scaled_progress,
                f"Extracting content from second PDF ({int(progress*100)}%)",
            )

        update_task_progress(task_id, 25, "Extracting content from second PDF")
        logger.info(f"Extracting content from {filename2}")
        # PDF content is already in memory
        pdf2_data = extractor.extract_pdf_content(pdf2_content, extraction2_progress)

        def comparison_progress(progress):
            scaled_progress = 45 + (progress * 30)
            update_task_progress(
                task_id, scaled_progress, f"Comparing PDFs ({int(progress*100)}%)"
            )

        update_task_progress(task_id, 45, "Comparing PDFs")
        logger.info("Comparing PDFs with optimized algorithm")
        comparer = PdfCompare(
            diff_threshold=0.75,
            cell_match_threshold=0.9,
            fuzzy_match_threshold=0.8,
            max_workers=os.cpu_count(),
        )
        comparison_results = comparer.compare_pdfs(
            pdf1_data, pdf2_data, comparison_progress
        )

        def report_progress(progress):
            scaled_progress = 75 + (progress * 23)
            update_task_progress(
                task_id,
                scaled_progress,
                f"Generating comparison report ({int(progress*100)}%)",
            )

        update_task_progress(task_id, 75, "Generating comparison report")
        logger.info("Generating enhanced comparison report")
        generator = ReportGenerator(output_dir=app_instance.config["REPORT_FOLDER"])
        report_path = generator.generate_html_report(
            comparison_results,
            filename1,
            filename2,
            metadata={
                "Original FileNames": f"{filename1}, {filename2}",
                "Comparison Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            progress_callback=report_progress,
        )

        report_filename = os.path.basename(report_path)
        summary_for_index_html = generator._calculate_summary(comparison_results)
        report_url = processing_tasks[task_id]["report_url_base"] + report_filename

        result = {
            "total_pages": comparison_results.get("max_pages", 0),
            "pages_with_differences": summary_for_index_html["pages_with_differences"],
            "text_differences": summary_for_index_html["text_differences"],
            "table_differences": summary_for_index_html["table_differences"],
            "total_differences": summary_for_index_html["text_differences"]
            + summary_for_index_html["table_differences"],
            "filename1": filename1,
            "filename2": filename2,
            "report_url": report_url,
            "report_filename": report_filename,
            "comparison_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        update_task_progress(task_id, 100, "Comparison completed")
        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["result"] = result

        logger.info(
            f"Comparison completed successfully for task {task_id}, report saved as {report_filename}"
        )
        # No uploaded files to clean up.

    except Exception as e:
        logger.error(f"Error in PDF comparison task {task_id}: {str(e)}", exc_info=True)
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)
        processing_tasks[task_id]["status_message"] = f"Error: {str(e)}"
        # No temporary files to clean up in case of an error.


def update_task_progress(task_id, progress, status_message=None):
    """Update task progress and status message atomically."""
    if task_id in processing_tasks:
        processing_tasks[task_id]["progress"] = progress
        if status_message:
            processing_tasks[task_id]["status_message"] = status_message
        logger.debug(f"Task {task_id} progress: {progress}%, {status_message}")


@app.route("/status/<task_id>")
def get_task_status(task_id):
    """Get the status of a processing task with enhanced details."""
    if task_id not in processing_tasks:
        return jsonify({"status": "error", "message": "Task not found"}), 404

    task = processing_tasks[task_id]
    response = {
        "status": task["status"],
        "progress": task["progress"],
        "status_message": task.get("status_message", ""),
        "result": task.get("result", None),
    }
    
    if "start_time" in task:
        response["start_time"] = task["start_time"]
        start_time = datetime.fromisoformat(task["start_time"])
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        response["elapsed_seconds"] = elapsed_seconds

    if task["status"] == "failed" and "error" in task:
        response["error"] = task["error"]

    return jsonify(response)


@app.route("/reports/<filename>")
def get_report(filename):
    """Serve generated comparison report."""
    try:
        return send_file(
            os.path.join(app.config["REPORT_FOLDER"], filename), mimetype="text/html"
        )
    except Exception as e:
        logger.error(f"Error serving report {filename}: {str(e)}", exc_info=True)
        return (
            jsonify({"status": "error", "message": f"Report not found: {str(e)}"}),
            404,
        )


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """API endpoint for PDF comparison using in-memory processing."""
    try:
        if "pdf1" not in request.files or "pdf2" not in request.files:
            return (
                jsonify({"status": "error", "message": "Both PDF files are required"}),
                400,
            )

        file1 = request.files["pdf1"]
        file2 = request.files["pdf2"]

        if file1.filename == "" or file2.filename == "":
            return (
                jsonify({"status": "error", "message": "File names cannot be empty"}),
                400,
            )

        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            return (
                jsonify({"status": "error", "message": "Only PDF files are allowed"}),
                400,
            )

        # Read content into memory; no temp files are created.
        pdf1_content = file1.read()
        pdf2_content = file2.read()
        
        orig_filename1 = secure_filename(file1.filename)
        orig_filename2 = secure_filename(file2.filename)
        
        task_id = str(uuid.uuid4())
        
        with app.app_context():
            report_url_template = url_for("get_report", filename="PLACEHOLDER")
            report_url_base = report_url_template.replace("PLACEHOLDER", "")

        processing_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "orig_filename1": orig_filename1,
            "orig_filename2": orig_filename2,
            "start_time": datetime.now().isoformat(),
            "is_api": True,
            "report_url_base": report_url_base,
            "status_message": "Initializing comparison",
        }

        threading.Thread(
            target=process_comparison_with_progress,
            args=(
                task_id,
                pdf1_content,
                pdf2_content,
                orig_filename1,
                orig_filename2,
                app,
            ),
        ).start()

        return jsonify(
            {
                "status": "processing",
                "message": "Comparison started",
                "task_id": task_id,
            }
        )

    except Exception as e:
        logger.error(f"API error: {str(e)}", exc_info=True)
        return (
            jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}),
            500,
        )


@app.route("/api/async-result/<task_id>")
def get_async_result(task_id):
    """Get result of asynchronous task with enhanced details."""
    if task_id not in processing_tasks:
        return jsonify({"status": "error", "message": "Task not found"}), 404

    task = processing_tasks[task_id]
    start_time = datetime.fromisoformat(task["start_time"])
    elapsed_seconds = (datetime.now() - start_time).total_seconds()

    if task["status"] == "completed":
        return jsonify(
            {
                "status": "success",
                "result": task.get("result"),
                "elapsed_seconds": elapsed_seconds,
            }
        )
    elif task["status"] == "failed":
        return (
            jsonify(
                {
                    "status": "error",
                    "message": task.get("error", "Unknown error"),
                    "elapsed_seconds": elapsed_seconds,
                }
            ),
            500,
        )
    else:
        return jsonify(
            {
                "status": "processing",
                "progress": task["progress"],
                "message": task.get("status_message", "Processing..."),
                "elapsed_seconds": elapsed_seconds,
            }
        )


@app.route("/api/cancel/<task_id>", methods=["POST"])
def cancel_task(task_id):
    """Cancel an in-progress comparison task."""
    if task_id not in processing_tasks:
        return jsonify({"status": "error", "message": "Task not found"}), 404

    task = processing_tasks[task_id]

    if task["status"] == "processing":
        task["status"] = "cancelled"
        task["status_message"] = "Task cancelled by user"
        # No file cleanup needed as files are not stored on disk
        logger.info(f"Task {task_id} cancelled by user.")
        return jsonify({"status": "success", "message": "Task cancelled successfully"})
    else:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f'Cannot cancel task with status: {task["status"]}',
                }
            ),
            400,
        )


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size exceeded error."""
    return (
        jsonify(
            {"status": "error", "message": f"File size exceeded maximum limit (64MB)"}
        ),
        413,
    )


@app.errorhandler(404)
def page_not_found(error):
    """Handle 404 errors."""
    return jsonify({"status": "error", "message": "Resource not found"}), 404


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return jsonify({"status": "error", "message": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Starting Enhanced PDF Comparison Tool v2.0")
    app.run(debug=True, host="0.0.0.0", port=8000)
