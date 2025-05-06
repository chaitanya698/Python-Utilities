"""
main.py - Enhanced PDF Comparison Tool
--------------------------------------
Integrates the improved modules for table detection and comparison with
parallel processing and progress tracking
"""
import os
import io
import logging
import tempfile
import traceback
import threading
import uuid
import json
import time
from flask import Flask, request, render_template, send_file, jsonify, redirect, url_for, current_app
from werkzeug.utils import secure_filename
from datetime import datetime

# Import our enhanced modules
from Extract import PDFExtractor
from compare import PdfCompare
from generate import ReportGenerator

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_compare.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['REPORT_FOLDER'] = os.path.join(os.getcwd(), 'reports')
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# In-memory store for task tracking
processing_tasks = {}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    """Enhanced file upload handler with progress reporting."""
    try:
        # Check if both files are present
        if 'pdf1' not in request.files or 'pdf2' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Both PDF files are required'
            }), 400
            
        file1 = request.files['pdf1']
        file2 = request.files['pdf2']
        
        # Check if filenames are empty
        if file1.filename == '' or file2.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'File names cannot be empty'
            }), 400
            
        # Check if files are allowed
        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            return jsonify({
                'status': 'error',
                'message': 'Only PDF files are allowed'
            }), 400
            
        # Save files with unique IDs to prevent filename collisions
        unique_id = str(uuid.uuid4())
        filename1 = f"{unique_id}_{secure_filename(file1.filename)}"
        filename2 = f"{unique_id}_{secure_filename(file2.filename)}"
        
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        
        file1.save(filepath1)
        file2.save(filepath2)
        
        logger.info(f"Files uploaded: {filename1}, {filename2}")
        
        # Create a task for tracking
        task_id = str(uuid.uuid4())
        
        # Create report URL in advance (within application context)
        # We'll use a placeholder for the report filename, which will be updated later
        with app.app_context():
            report_url_template = url_for('get_report', filename='PLACEHOLDER')
            report_url_base = report_url_template.replace('PLACEHOLDER', '')
        
        # Store task info in memory
        processing_tasks[task_id] = {
            'status': 'processing',
            'progress': 0,
            'filepath1': filepath1,
            'filepath2': filepath2,
            'orig_filename1': file1.filename,
            'orig_filename2': file2.filename,
            'start_time': datetime.now().isoformat(),
            'report_url_base': report_url_base  # Store the base URL for the report
        }
        
        # Start background processing
        threading.Thread(
            target=process_comparison_with_progress,
            args=(task_id, filepath1, filepath2, file1.filename, file2.filename, app)
        ).start()
        
        return jsonify({
            'status': 'processing',
            'message': 'Comparison started',
            'task_id': task_id
        })
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }), 500


def process_comparison_with_progress(task_id, filepath1, filepath2, filename1, filename2, app_instance):
    """Process PDF comparison with progress tracking."""
    try:
        logger.info(f"Starting comparison task {task_id} between {filename1} and {filename2}")
        
        # Update task progress
        processing_tasks[task_id]['progress'] = 5
        processing_tasks[task_id]['status_message'] = 'Extracting content from first PDF'
        
        # Read file contents
        with open(filepath1, 'rb') as f1, open(filepath2, 'rb') as f2:
            pdf1_content = f1.read()
            pdf2_content = f2.read()
        
        # Extract content from PDFs with enhanced table detection
        extractor = PDFExtractor(
            similarity_threshold=0.85,  # Threshold for table content similarity
            header_match_threshold=0.9,  # Threshold for header matching
            nested_table_threshold=0.85,  # Containment threshold for nested tables
            nested_area_ratio=0.75      # Size ratio for nested tables
        )
        logger.info(f"Extracting content from {filename1}")
        pdf1_data = extractor.extract_pdf_content(pdf1_content)
        
        # Update task progress
        processing_tasks[task_id]['progress'] = 25
        processing_tasks[task_id]['status_message'] = 'Extracting content from second PDF'
        
        logger.info(f"Extracting content from {filename2}")
        pdf2_data = extractor.extract_pdf_content(pdf2_content)
        
        # Update task progress
        processing_tasks[task_id]['progress'] = 50
        processing_tasks[task_id]['status_message'] = 'Comparing PDFs'
        
        # Compare PDFs with improved matching algorithms
        logger.info("Comparing PDFs")
        comparer = PdfCompare(
            diff_threshold=0.75,             # Threshold for table similarity
            cell_match_threshold=0.9,        # Threshold for cell content matching
            fuzzy_match_threshold=0.8,       # Threshold for fuzzy text matching
            max_workers=4                    # Parallel processing threads
        )
        comparison_results = comparer.compare_pdfs(pdf1_data, pdf2_data)
        
        # Update task progress
        processing_tasks[task_id]['progress'] = 75
        processing_tasks[task_id]['status_message'] = 'Generating comparison report'
        
        # Generate enhanced HTML report
        logger.info("Generating comparison report")
        generator = ReportGenerator(output_dir=app_instance.config['REPORT_FOLDER'])
        report_path = generator.generate_html_report(
            comparison_results,
            filename1,
            filename2,
            metadata={
                'Original Filenames': f"{filename1}, {filename2}",
                'Comparison Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Tool Version': '2.0'
            }
        )
        
        # Generate response
        report_filename = os.path.basename(report_path)
        
        # Generate summary statistics
        summary = calculate_summary(comparison_results)
        
        # Create report URL without using url_for() outside app context
        report_url = processing_tasks[task_id]['report_url_base'] + report_filename
        
        # Prepare detailed response data
        result = {
            'total_pages': comparison_results.get('max_pages', 0),
            'pages_with_differences': summary['pages_with_differences'],
            'total_differences': summary['total_differences'],
            'text_differences': summary['text_differences'],
            'table_differences': summary['table_differences'],
            'filename1': filename1,
            'filename2': filename2,
            'report_url': report_url,
            'report_filename': report_filename,
            'comparison_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Update task as completed
        processing_tasks[task_id]['progress'] = 100
        processing_tasks[task_id]['status'] = 'completed'
        processing_tasks[task_id]['result'] = result
        processing_tasks[task_id]['status_message'] = 'Comparison completed'
        
        logger.info(f"Comparison completed successfully for task {task_id}, report saved as {report_filename}")
        
        # Clean up temporary files after an hour
        schedule_cleanup(filepath1, filepath2, 3600)  # 1 hour
        
    except Exception as e:
        logger.error(f"Error in PDF comparison task {task_id}: {str(e)}", exc_info=True)
        
        # Update task as failed
        processing_tasks[task_id]['status'] = 'failed'
        processing_tasks[task_id]['error'] = str(e)
        processing_tasks[task_id]['status_message'] = f'Error: {str(e)}'
        
        # Clean up immediately in case of error
        try:
            os.remove(filepath1)
            os.remove(filepath2)
        except:
            pass


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


def calculate_summary(comparison_results):
    """Calculate comprehensive summary statistics for comparison results."""
    total_pages = comparison_results.get('max_pages', 0)
    pages_with_differences = 0
    total_text_differences = 0
    total_table_differences = 0
    
    # Count differences by page
    for page_num, page_data in comparison_results.get('pages', {}).items():
        page_has_differences = False
        
        # Count text differences
        text_diffs = [d for d in page_data.get('text_differences', []) 
                     if d.get('status') != 'equal']
        if text_diffs:
            page_has_differences = True
            total_text_differences += len(text_diffs)
        
        # Count table differences - excluding 'matched' and 'moved' statuses
        table_diffs = [t for t in page_data.get('table_differences', []) 
                      if t.get('status') not in ('matched', 'moved')]
        if table_diffs:
            page_has_differences = True
            total_table_differences += len(table_diffs)
        
        if page_has_differences:
            pages_with_differences += 1
    
    # Calculate percentage changed
    percentage_changed = round((pages_with_differences / total_pages) * 100, 2) if total_pages > 0 else 0
    
    return {
        'total_pages': total_pages,
        'pages_with_differences': pages_with_differences,
        'percentage_changed': percentage_changed,
        'text_differences': total_text_differences,
        'table_differences': total_table_differences,
        'total_differences': total_text_differences + total_table_differences
    }


@app.route('/status/<task_id>')
def get_task_status(task_id):
    """Get the status of a processing task."""
    if task_id not in processing_tasks:
        return jsonify({
            'status': 'error',
            'message': 'Task not found'
        }), 404
    
    task = processing_tasks[task_id]
    
    return jsonify({
        'status': task['status'],
        'progress': task['progress'],
        'status_message': task.get('status_message', ''),
        'result': task.get('result', None)
    })


@app.route('/reports/<filename>')
def get_report(filename):
    """Serve generated comparison report."""
    try:
        return send_file(
            os.path.join(app.config['REPORT_FOLDER'], filename),
            mimetype='text/html'
        )
    except Exception as e:
        logger.error(f"Error serving report {filename}: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Report not found: {str(e)}'
        }), 404


@app.route('/api/compare', methods=['POST'])
def api_compare():
    """
    API endpoint for PDF comparison.
    
    Expects two PDF files to be uploaded: 'pdf1' and 'pdf2'.
    Returns JSON with task ID for status tracking.
    """
    try:
        # Check if both files are present
        if 'pdf1' not in request.files or 'pdf2' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Both PDF files are required'
            }), 400
            
        file1 = request.files['pdf1']
        file2 = request.files['pdf2']
        
        # Check if filenames are empty
        if file1.filename == '' or file2.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'File names cannot be empty'
            }), 400
            
        # Check if files are allowed
        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            return jsonify({
                'status': 'error',
                'message': 'Only PDF files are allowed'
            }), 400
            
        # Create unique task ID
        task_id = str(uuid.uuid4())
        
        # Create report URL in advance (within application context)
        with app.app_context():
            report_url_template = url_for('get_report', filename='PLACEHOLDER')
            report_url_base = report_url_template.replace('PLACEHOLDER', '')
            
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp1, \
             tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp2:
            
            file1.save(temp1.name)
            file2.save(temp2.name)
            
            # Store task info
            processing_tasks[task_id] = {
                'status': 'processing',
                'progress': 0,
                'filepath1': temp1.name,
                'filepath2': temp2.name,
                'orig_filename1': secure_filename(file1.filename),
                'orig_filename2': secure_filename(file2.filename),
                'start_time': datetime.now().isoformat(),
                'is_api': True,  # Mark as API task
                'report_url_base': report_url_base  # Store the base URL for the report
            }
            
            # Start background processing
            threading.Thread(
                target=process_comparison_with_progress,
                args=(task_id, temp1.name, temp2.name, file1.filename, file2.filename, app)
            ).start()
            
            return jsonify({
                'status': 'processing',
                'message': 'Comparison started',
                'task_id': task_id
            })
            
    except Exception as e:
        logger.error(f"API error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }), 500


@app.route('/api/async-result/<task_id>')
def get_async_result(task_id):
    """Get result of asynchronous task."""
    if task_id not in processing_tasks:
        return jsonify({
            'status': 'error',
            'message': 'Task not found'
        }), 404
    
    task = processing_tasks[task_id]
    
    if task['status'] == 'completed':
        return jsonify({
            'status': 'success',
            'result': task.get('result')
        })
    elif task['status'] == 'failed':
        return jsonify({
            'status': 'error',
            'message': task.get('error', 'Unknown error')
        }), 500
    else:
        return jsonify({
            'status': 'processing',
            'progress': task['progress'],
            'message': task.get('status_message', 'Processing...')
        })


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size exceeded error."""
    return jsonify({
        'status': 'error',
        'message': f'File size exceeded maximum limit (64MB)'
    }), 413


@app.errorhandler(404)
def page_not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'status': 'error',
        'message': 'Resource not found'
    }), 404


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500


if __name__ == '__main__':
    logger.info("Starting PDF Comparison Tool v2.0")
    app.run(debug=True, host='0.0.0.0', port=8000)