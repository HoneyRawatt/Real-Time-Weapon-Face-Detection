import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os

sender = os.getenv('LOCAL_TEST_SENDER', 'test@example.com')
recipient = os.getenv('LOCAL_TEST_TO', 'rawathoney639@gmail.com')
image_path = os.getenv('LOCAL_TEST_IMAGE', None)

subject = 'Local Debug: Security Alert - Test'
body = 'This is a test email sent to the local SMTP debug server.'

msg = MIMEMultipart()
msg['From'] = sender
msg['To'] = recipient
msg['Subject'] = subject
msg.attach(MIMEText(body, 'plain'))

if image_path and os.path.exists(image_path):
    with open(image_path, 'rb') as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(image_path)}')
        msg.attach(part)
else:
    print('No image attached for this test.')

print('Connecting to local SMTP (localhost:1025)')
try:
    server = smtplib.SMTP('localhost', 1025)
    server.send_message(msg)
    server.quit()
    print('Local test email sent successfully (no auth required).')
except Exception as e:
    print('Failed to send local test email:', e)
