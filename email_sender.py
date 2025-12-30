import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os

def send_email_with_attachment(image_path, to_email='rawathoney639@gmail.com'):
    sender_email = 'projecttrain12@gmail.com'
    password = 'project 1234'

    subject = 'Security Alert: Unknown Person Detected'
    body = 'An unknown person with a weapon has been detected. Please find the attached image.'

    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    
    if os.path.exists(image_path):
        with open(image_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(image_path)}')
            msg.attach(part)
    else:
        print('Image not found, sending email without attachment.')

    
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        print('Email sent successfully.')
    except Exception as e:
        print(f'Failed to send email: {e}')
