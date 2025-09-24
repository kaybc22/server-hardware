import http.server
import socketserver
import datetime
import os

PORT = 8000
REFRESH_INTERVAL_SECONDS = 3 # How often the browser will refresh (e.g., every 3 seconds)
LINES_TO_TAIL = 40        # Number of lines to display from the end of each file

# List of files to monitor (ensure they are in the same directory as the script)
MONITORED_FILES = [
    "nv_gpu_usage.log",
    "bmc_info_20250723_211943.log"
]

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            html_content = self.generate_html_dashboard()
            self.wfile.write(html_content.encode('utf-8'))
        else:
            # Serve other files normally if requested
            super().do_GET()

    def generate_html_dashboard(self):
        # HTML header with auto-refresh and CSS for side-by-side layout
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="X-UA-Compatible" content="IE=edge">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <meta http-equiv="refresh" content="{REFRESH_INTERVAL_SECONDS}">
            <title>Real-time Data Monitor</title>
            <style>
                body {{
                    font-family: monospace;
                    background-color: #1e1e1e;
                    color: #d4d4d4;
                    margin: 20px;
                    display: flex;
                    flex-direction: column;
                    min-height: 95vh; /* Ensure body takes enough height for columns to scale */
                }}
                h1 {{ color: #569cd6; }}
                h2 {{ color: #4ec9b0; border-bottom: 1px solid #4ec9b0; padding-bottom: 5px; margin-top: 20px; }}
                .timestamp {{ color: #9cdcfe; font-size: 0.9em; margin-bottom: 20px; }}

                .container {{
                    display: flex; /* Use Flexbox for side-by-side layout */
                    gap: 20px;     /* Space between columns */
                    flex-wrap: wrap; /* Allow columns to wrap to next line on smaller screens */
                    flex-grow: 1; /* Allow container to grow and fill available space */
                }}
                .file-section {{
                    flex: 1; /* Each section takes equal available space */
                    min-width: 300px; /* Minimum width for each column before wrapping */
                    background-color: #252526;
                    padding: 10px;
                    border-radius: 5px;
                    overflow-y: auto; /* Enable vertical scrolling within each column */
                    max-height: calc(100vh - 180px); /* Adjust based on header/footer size to fit viewport */
                    display: flex; /* Inner flex for title and pre */
                    flex-direction: column;
                }}
                .file-section h2 {{
                    flex-shrink: 0; /* Prevent title from shrinking */
                    margin-top: 0;
                }}
                .file-section pre {{
                    flex-grow: 1; /* Preformatted text area grows to fill available height */
                    margin: 0;
                    white-space: pre-wrap; /* Wrap long lines */
                    word-wrap: break-word; /* Break words if necessary */
                    overflow-x: hidden; /* Hide horizontal scroll within pre */
                }}
            </style>
        </head>
        <body>
            <h1>Real-time Data Dashboard</h1>
            <p class="timestamp">Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <div class="container">
        """

        # Read and append content from each file
        for filename in MONITORED_FILES:
            html += f'<div class="file-section"><h2>{filename} (Last {LINES_TO_TAIL} lines)</h2>'
            try:
                with open(filename, 'r') as f:
                    lines = f.readlines()
                    # Get the last LINES_TO_TAIL lines
                    content = "".join(lines[-LINES_TO_TAIL:])
                    html += f'<pre>{content}</pre>'
            except FileNotFoundError:
                html += f'<pre>Error: File "{filename}" not found.</pre>'
            except Exception as e:
                html += f'<pre>Error reading "{filename}": {e}</pre>'
            html += '</div>'

        html += """
            </div>
        </body>
        </html>
        """
        return html

# Start the server
with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
    print(f"Serving real-time dashboard at http://localhost:{PORT}")
    print(f"Dashboard will refresh every {REFRESH_INTERVAL_SECONDS} seconds.")
    print(f"Displaying last {LINES_TO_TAIL} lines of each file.")
    print("Press Ctrl+C to stop the server.")
    httpd.serve_forever()