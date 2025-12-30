import os
import sys
# Ensure project root is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from email_sender import send_email_with_attachment_with_credentials

# Read credentials from environment for safety
sender = os.getenv('TEST_SENDER_EMAIL')
password = os.getenv('TEST_EMAIL_PASSWORD')
recipient = os.getenv('TEST_TO_EMAIL', 'rawathoney639@gmail.com')
image = os.getenv('TEST_IMAGE_PATH', None)  # optional

if not sender or not password:
    print('ERROR: Set TEST_SENDER_EMAIL and TEST_EMAIL_PASSWORD environment variables to run this test.')
    print('Example (PowerShell):')
    print("$env:TEST_SENDER_EMAIL='youremail@gmail.com'; $env:TEST_EMAIL_PASSWORD='your_app_password'; $env:TEST_TO_EMAIL='recipient@gmail.com'; & '.\\.venv311\\Scripts\\python.exe' scripts/send_test_email_with_credentials.py")
    sys.exit(1)

print(f"Testing email send\n  From: {sender}\n  To:   {recipient}\n  Image: {image or '(none)'}")

ok = send_email_with_attachment_with_credentials(image, sender, password, recipient)
if ok:
    print('Test succeeded.')
else:
    print('Test failed.')
