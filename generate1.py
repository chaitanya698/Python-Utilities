import os
import logging
import tempfile
import threading
import uuid
import time
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
app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "uploads")
app.config["REPORT_FOLDER"] = os.path.join(os.getcwd(), "reports")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64MB max upload size
app.config["ALLOWED_EXTENSIONS"] = {"pdf"}

# In-memory store for task tracking
processing_tasks = {}

# Create necessary directories
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
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
    """Enhanced file upload handler with progress reporting."""
    try:
        # Check if both files are present
        if "pdf1" not in request.files or "pdf2" not in request.files:
            return (
                jsonify({"status": "error", "message": "Both PDF files are required"}),
                400,
            )

        file1 = request.files["pdf1"]
        file2 = request.files["pdf2"]

        # Check if filenames are empty
        if file1.filename == "" or file2.filename == "":
            return (
                jsonify({"status": "error", "message": "File names cannot be empty"}),
                400,
            )

        # Check if files are allowed
        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            return (
                jsonify({"status": "error", "message": "Only PDF files are allowed"}),
                400,
            )

        # Save files with unique IDs to prevent filename collisions
        unique_id = str(uuid.uuid4())
        filename1 = f"{unique_id}_{secure_filename(file1.filename)}"
        filename2 = f"{unique_id}_{secure_filename(file2.filename)}"

        filepath1 = os.path.join(app.config["UPLOAD_FOLDER"], filename1)
        filepath2 = os.path.join(app.config["UPLOAD_FOLDER"], filename2)

        file1.save(filepath1)
        file2.save(filepath2)

        logger.info(f"Files uploaded: {filename1}, {filename2}")

        # Create a task for tracking
        task_id = str(uuid.uuid4())

        # Create report URL in advance (within application context)
        # We'll use a placeholder for the report filename, which will be updated later
        with app.app_context():
            report_url_template = url_for("get_report", filename="PLACEHOLDER")
            report_url_base = report_url_template.replace("PLACEHOLDER", "")

        # Store task info in memory
        processing_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "filepath1": filepath1,
            "filepath2": filepath2,
            "orig_filename1": file1.filename,
            "orig_filename2": file2.filename,
            "start_time": datetime.now().isoformat(),
            "report_url_base": report_url_base,  # Store the base URL for the report
            "status_message": "Initializing comparison",
        }

        # Start background processing
        threading.Thread(
            target=process_comparison_with_progress,
            args=(task_id, filepath1, filepath2, file1.filename, file2.filename, app),
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
    task_id, filepath1, filepath2, filename1, filename2, app_instance
):
    """
    Process PDF comparison with progress tracking.
    Implements the optimized workflow with enhanced modules.
    """
    try:
        logger.info(
            f"Starting comparison task {task_id} between {filename1} and {filename2}"
        )

        # Progress callback for extraction phase
        def extraction_progress(progress):
            # Scale progress to 0-25% for first extraction
            scaled_progress = 5 + (progress * 20)
            update_task_progress(
                task_id,
                scaled_progress,
                f"Extracting content from first PDF ({int(progress*100)}%)",
            )

        # Read first file contents
        with open(filepath1, "rb") as f1:
            pdf1_content = f1.read()

        # Extract content from first PDF with optimized extractor
        extractor = PDFExtractor(
            similarity_threshold=0.85,  # Threshold for table content similarity
            header_match_threshold=0.9,  # Threshold for header matching
            nested_table_threshold=0.85,  # Containment threshold for nested tables
            nested_area_ratio=0.75,  # Size ratio for nested tables
            max_workers=os.cpu_count(),  # Use all available cores
        )

        update_task_progress(task_id, 5, "Extracting content from first PDF")
        logger.info(f"Extracting content from {filename1}")
        pdf1_data = extractor.extract_pdf_content(pdf1_content, extraction_progress)

        # Progress callback for second extraction
        def extraction2_progress(progress):
            # Scale progress to 25-45% for second extraction
            scaled_progress = 25 + (progress * 20)
            update_task_progress(
                task_id,
                scaled_progress,
                f"Extracting content from second PDF ({int(progress*100)}%)",
            )

        # Read second file contents
        with open(filepath2, "rb") as f2:
            pdf2_content = f2.read()

        # Extract content from second PDF
        update_task_progress(task_id, 25, "Extracting content from second PDF")
        logger.info(f"Extracting content from {filename2}")
        pdf2_data = extractor.extract_pdf_content(pdf2_content, extraction2_progress)

        # Progress callback for comparison
        def comparison_progress(progress):
            # Scale progress to 45-75% for comparison
            scaled_progress = 45 + (progress * 30)
            update_task_progress(
                task_id, scaled_progress, f"Comparing PDFs ({int(progress*100)}%)"
            )

        # Compare PDFs with optimized comparison algorithm
        update_task_progress(task_id, 45, "Comparing PDFs")
        logger.info("Comparing PDFs with optimized algorithm")
        comparer = PdfCompare(
            diff_threshold=0.75,  # Threshold for table similarity
            cell_match_threshold=0.9,  # Threshold for cell content matching
            fuzzy_match_threshold=0.8,  # Threshold for fuzzy text matching
            max_workers=os.cpu_count(),  # Use all available cores
        )
        comparison_results = comparer.compare_pdfs(
            pdf1_data, pdf2_data, comparison_progress
        )

        # Progress callback for report generation
        def report_progress(progress):
            # Scale progress to 75-98% for report generation
            scaled_progress = 75 + (progress * 23)
            update_task_progress(
                task_id,
                scaled_progress,
                f"Generating comparison report ({int(progress*100)}%)",
            )

        # Generate enhanced HTML report
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

        # Generate response
        report_filename = os.path.basename(report_path)

        # Generate summary_for_index_html statistics
        summary_for_index_html = generator._calculate_summary(comparison_results)

        # Create report URL
        report_url = processing_tasks[task_id]["report_url_base"] + report_filename

        # Prepare detailed response data
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

        # Update task as completed
        update_task_progress(task_id, 100, "Comparison completed")
        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["result"] = result

        logger.info(
            f"Comparison completed successfully for task {task_id}, report saved as {report_filename}"
        )

        # Clean up temporary files after an hour
        schedule_cleanup(filepath1, filepath2, 3600)  # 1 hour

    except Exception as e:
        logger.error(f"Error in PDF comparison task {task_id}: {str(e)}", exc_info=True)

        # Update task as failed
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)
        processing_tasks[task_id]["status_message"] = f"Error: {str(e)}"

        # Clean up immediately in case of error
        try:
            os.remove(filepath1)
            os.remove(filepath2)
        except:
            pass


def update_task_progress(task_id, progress, status_message=None):
    """Update task progress and status message atomically."""
    if task_id in processing_tasks:
        processing_tasks[task_id]["progress"] = progress
        if status_message:
            processing_tasks[task_id]["status_message"] = status_message
        logger.debug(f"Task {task_id} progress: {progress}%, {status_message}")


def schedule_cleanup(file1, file2, delay):
    """Schedule cleanup of temporary files after a delay."""

    def cleanup():
        time.sleep(delay)
        try:
            if os.path.exists(file1):
                os.remove(file1)
            if os.path.exists(file2):
                os.remove(file2)
            logger.info(f"Cleaned up temporary files: {file1}, {file2}")
        except Exception as e:
            logger.error(f"Error cleaning up files: {str(e)}")

    threading.Thread(target=cleanup).start()


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

    # Add start time and elapsed time for better visibility
    if "start_time" in task:
        response["start_time"] = task["start_time"]
        # Calculate elapsed time
        start_time = datetime.fromisoformat(task["start_time"])
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        response["elapsed_seconds"] = elapsed_seconds

    # Add error details if available
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
    """
    API endpoint for PDF comparison.

    Expects two PDF files to be uploaded: 'pdf1' and 'pdf2'.
    Returns JSON with task ID for status tracking.
    """
    try:
        # Check if both files are present
        if "pdf1" not in request.files or "pdf2" not in request.files:
            return (
                jsonify({"status": "error", "message": "Both PDF files are required"}),
                400,
            )

        file1 = request.files["pdf1"]
        file2 = request.files["pdf2"]

        # Check if filenames are empty
        if file1.filename == "" or file2.filename == "":
            return (
                jsonify({"status": "error", "message": "File names cannot be empty"}),
                400,
            )

        # Check if files are allowed
        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            return (
                jsonify({"status": "error", "message": "Only PDF files are allowed"}),
                400,
            )

        # Create unique task ID
        task_id = str(uuid.uuid4())

        # Create report URL in advance (within application context)
        with app.app_context():
            report_url_template = url_for("get_report", filename="PLACEHOLDER")
            report_url_base = report_url_template.replace("PLACEHOLDER", "")

        # Create temporary files
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False
        ) as temp1, tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp2:

            file1.save(temp1.name)
            file2.save(temp2.name)

            # Store task info
            processing_tasks[task_id] = {
                "status": "processing",
                "progress": 0,
                "filepath1": temp1.name,
                "filepath2": temp2.name,
                "orig_filename1": secure_filename(file1.filename),
                "orig_filename2": secure_filename(file2.filename),
                "start_time": datetime.now().isoformat(),
                "is_api": True,  # Mark as API task
                "report_url_base": report_url_base,  # Store the base URL for the report
                "status_message": "Initializing comparison",
            }

            # Start background processing
            threading.Thread(
                target=process_comparison_with_progress,
                args=(
                    task_id,
                    temp1.name,
                    temp2.name,
                    file1.filename,
                    file2.filename,
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
    """
    Get result of asynchronous task with enhanced details.
    Returns more detailed progress and error information.
    """
    if task_id not in processing_tasks:
        return jsonify({"status": "error", "message": "Task not found"}), 404

    task = processing_tasks[task_id]

    # Calculate elapsed time
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

    # Only cancel processing tasks
    if task["status"] == "processing":
        # Mark as cancelled
        task["status"] = "cancelled"
        task["status_message"] = "Task cancelled by user"

        # Attempt to clean up files
        try:
            if os.path.exists(task["filepath1"]):
                os.remove(task["filepath1"])
            if os.path.exists(task["filepath2"]):
                os.remove(task["filepath2"])
        except Exception as e:
            logger.error(f"Error cleaning up files for cancelled task: {str(e)}")

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
