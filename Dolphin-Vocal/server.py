from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import numpy as np
from edge_impulse_classifier import EdgeImpulseClassifier  # Import your Edge Impulse classifier module

PORT = 8082

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        if self.path == '/classify-audio':
            # Perform audio classification using the Edge Impulse classifier
            classification_result = classify_audio(post_data)

            # Respond with the classification result
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(classification_result).encode())
        else:
            self.send_response(404)
            self.end_headers()

def classify_audio(audio_data):
    # Initialize Edge Impulse Classifier
    classifier = EdgeImpulseClassifier()

    # Classify audio using the Edge Impulse model
    classification_result = classifier.classify_audio(audio_data)

    return classification_result

def run_server():
    httpd = HTTPServer(('localhost', PORT), RequestHandler)
    print(f'Server running on localhost:{PORT}')
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
