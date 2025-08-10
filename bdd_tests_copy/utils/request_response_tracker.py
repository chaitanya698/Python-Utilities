import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import html


class RequestResponseTracker:
    """Track all API requests and responses for detailed reporting."""
    
    def __init__(self):
        self.history: Dict[str, List[Dict[str, Any]]] = {}
        self.current_test_id: Optional[str] = None
    
    def set_current_test(self, test_id: str):
        """Set the current test being executed."""
        self.current_test_id = test_id
        if test_id not in self.history:
            self.history[test_id] = []
    
    def add_request(self, method: str, url: str, headers: Dict, data: Any, correlation_id: str):
        """Add a request to the history."""
        if not self.current_test_id:
            return
        
        request_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'request',
            'method': method,
            'url': url,
            'correlation_id': correlation_id,
            'headers': self._sanitize_headers(headers),
            'data': data
        }
        
        self.history[self.current_test_id].append(request_entry)
    
    def add_response(self, status_code: int, headers: Dict, data: Any, duration: float, correlation_id: str):
        """Add a response to the history."""
        if not self.current_test_id:
            return
        
        response_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'response',
            'status_code': status_code,
            'correlation_id': correlation_id,
            'headers': dict(headers) if headers else {},
            'data': data,
            'duration_ms': round(duration * 1000, 2)
        }
        
        self.history[self.current_test_id].append(response_entry)
    
    def add_error(self, error_type: str, error_message: str, correlation_id: str):
        """Add an error to the history."""
        if not self.current_test_id:
            return
        
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'error',
            'correlation_id': correlation_id,
            'error_type': error_type,
            'error_message': error_message
        }
        
        self.history[self.current_test_id].append(error_entry)
    
    def get_test_history(self, test_id: str) -> List[Dict[str, Any]]:
        """Get the request/response history for a specific test."""
        return self.history.get(test_id, [])
    
    def _sanitize_headers(self, headers: Dict) -> Dict:
        """Remove sensitive information from headers."""
        sanitized = {}
        sensitive_keys = ['authorization', 'x-api-key', 'cookie', 'password', 'token']
        
        for key, value in headers.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = '***REDACTED***'
            else:
                sanitized[key] = value
        
        return sanitized
    
    def format_history_as_html(self, history: List[Dict[str, Any]]) -> str:
        """Format request/response history as HTML for the report."""
        if not history:
            return ""
        
        html_parts = ['<div class="api-history">']
        
        # Group requests and responses by correlation ID
        grouped = {}
        for entry in history:
            corr_id = entry.get('correlation_id', 'unknown')
            if corr_id not in grouped:
                grouped[corr_id] = []
            grouped[corr_id].append(entry)
        
        for corr_id, entries in grouped.items():
            html_parts.append(f'<div class="api-transaction">')
            html_parts.append(f'<h4>Transaction: {corr_id}</h4>')
            
            for entry in entries:
                if entry['type'] == 'request':
                    html_parts.append(self._format_request_html(entry))
                elif entry['type'] == 'response':
                    html_parts.append(self._format_response_html(entry))
                elif entry['type'] == 'error':
                    html_parts.append(self._format_error_html(entry))
            
            html_parts.append('</div>')
        
        html_parts.append('</div>')
        
        # Add CSS for styling
        css = """
        <style>
        .api-history { margin: 20px 0; }
        .api-transaction { 
            border: 1px solid #ddd; 
            margin: 10px 0; 
            padding: 15px; 
            border-radius: 5px;
            background: #f9f9f9;
        }
        .api-request, .api-response, .api-error { 
            margin: 10px 0; 
            padding: 10px; 
            border-radius: 3px;
        }
        .api-request { background: #e3f2fd; border-left: 4px solid #2196f3; }
        .api-response { background: #e8f5e9; border-left: 4px solid #4caf50; }
        .api-error { background: #ffebee; border-left: 4px solid #f44336; }
        .api-meta { color: #666; font-size: 0.9em; margin-bottom: 5px; }
        .api-data { 
            background: white; 
            padding: 10px; 
            border: 1px solid #ddd; 
            border-radius: 3px;
            font-family: monospace;
            font-size: 0.85em;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 300px;
            overflow-y: auto;
        }
        </style>
        """
        
        return css + '\n'.join(html_parts)
    
    def _format_request_html(self, entry: Dict) -> str:
        """Format a request entry as HTML."""
        data_str = json.dumps(entry.get('data'), indent=2, default=str) if entry.get('data') else 'No data'
        
        return f"""
        <div class="api-request">
            <div class="api-meta">
                <strong>REQUEST</strong> | {entry['timestamp']} | {entry['method']} {entry['url']}
            </div>
            <div class="api-data">{html.escape(data_str)}</div>
        </div>
        """
    
    def _format_response_html(self, entry: Dict) -> str:
        """Format a response entry as HTML."""
        data_str = json.dumps(entry.get('data'), indent=2, default=str) if entry.get('data') else 'No data'
        
        return f"""
        <div class="api-response">
            <div class="api-meta">
                <strong>RESPONSE</strong> | {entry['timestamp']} | 
                Status: {entry['status_code']} | Duration: {entry.get('duration_ms', 'N/A')}ms
            </div>
            <div class="api-data">{html.escape(data_str)}</div>
        </div>
        """
    
    def _format_error_html(self, entry: Dict) -> str:
        """Format an error entry as HTML."""
        return f"""
        <div class="api-error">
            <div class="api-meta">
                <strong>ERROR</strong> | {entry['timestamp']} | {entry['error_type']}
            </div>
            <div class="api-data">{html.escape(entry['error_message'])}</div>
        </div>
        """
