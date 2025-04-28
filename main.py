"""
main.py - Enhanced PDF Comparison Tool
--------------------------------------
Integrates the improved modules for table detection and comparison
"""
import os
import io
import logging
import tempfile
import traceback
from flask import Flask, request, render_template, send_file, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from datetime import datetime

# Import our enhanced modules
# Note: Update these imports to match your folder structure
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
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

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
    """Handle file upload and start comparison process."""
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
            
        # Save files with secure filenames
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        
        file1.save(filepath1)
        file2.save(filepath2)
        
        logger.info(f"Files uploaded: {filename1}, {filename2}")
        
        # Process comparison
        result = process_comparison(filepath1, filepath2, filename1, filename2)
        
        return jsonify({
            'status': 'success',
            'message': 'Comparison completed',
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }), 500


def process_comparison(filepath1, filepath2, filename1, filename2):
    """
    Process PDF comparison between two files with enhanced detection.
    
    Args:
        filepath1: Path to first PDF
        filepath2: Path to second PDF
        filename1: Original name of first PDF
        filename2: Original name of second PDF
        
    Returns:
        Dictionary with comparison results and report metadata
    """
    try:
        logger.info(f"Starting comparison between {filename1} and {filename2}")
        
        # Read file contents
        with open(filepath1, 'rb') as f1, open(filepath2, 'rb') as f2:
            pdf1_content = f1.read()
            pdf2_content = f2.read()
        
        # Extract content from PDFs with enhanced table detection
        extractor = PDFExtractor(
            similarity_threshold=0.85,  # Threshold for table content similarity
            header_match_threshold=0.9,  # Threshold for header matching
            nested_table_threshold=0.85,  # Containment threshold for nested tables
            nested_area_ratio=0.75       # Size ratio for nested tables
        )
        
        logger.info(f"Extracting content from {filename1}")
        pdf1_data = extractor.extract_pdf_content(pdf1_content)
        
        logger.info(f"Extracting content from {filename2}")
        pdf2_data = extractor.extract_pdf_content(pdf2_content)
        
        # Compare PDFs with improved matching algorithms
        logger.info("Comparing PDFs")
        comparer = PdfCompare(
            diff_threshold=0.75,  # Threshold for table similarity
            cell_match_threshold=0.9,  # Threshold for cell content matching
            fuzzy_match_threshold=0.8  # Threshold for fuzzy text matching
        )
        comparison_results = comparer.compare_pdfs(pdf1_data, pdf2_data)
        
        # Generate enhanced HTML report
        logger.info("Generating comparison report")
        generator = ReportGenerator(output_dir=app.config['REPORT_FOLDER'])
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
        
        # Prepare detailed response data
        result = {
            'total_pages': comparison_results.get('max_pages', 0),
            'pages_with_differences': summary['pages_with_differences'],
            'total_differences': summary['total_differences'],
            'text_differences': summary['text_differences'],
            'table_differences': summary['table_differences'],
            'filename1': filename1,
            'filename2': filename2,
            'report_url': url_for('get_report', filename=report_filename),
            'report_filename': report_filename,
            'comparison_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"Comparison completed successfully, report saved as {report_filename}")
        return result
        
    except Exception as e:
        logger.error(f"Error in PDF comparison: {str(e)}", exc_info=True)
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        raise Exception(f"PDF comparison failed: {str(e)}") from e


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
    Returns JSON with comparison results.
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
            
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp1, \
             tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp2:
            
            file1.save(temp1.name)
            file2.save(temp2.name)
            
            # Process files
            result = process_comparison(
                temp1.name,
                temp2.name,
                secure_filename(file1.filename),
                secure_filename(file2.filename)
            )
            
            # Clean up temporary files
            os.unlink(temp1.name)
            os.unlink(temp2.name)
            
            return jsonify({
                'status': 'success',
                'result': result
            })
            
    except Exception as e:
        logger.error(f"API error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size exceeded error."""
    return jsonify({
        'status': 'error',
        'message': f'File size exceeded maximum limit (32MB)'
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