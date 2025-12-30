import os
import sys

# Add parent directory to path so we can import email_sender
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from email_sender import send_email_with_attachment

# Use env vars set by caller (or use defaults if not set)
sender = os.environ.get('ALERT_SENDER_EMAIL', 'not-set')
to_email = os.environ.get('ALERT_TO_EMAIL', 'not-set')

print(f'📧 Sending test email')
print(f'   From: {sender}')
print(f'   To: {to_email}')
print()
try:
    send_email_with_attachment(None)
    print('✅ send_email_with_attachment completed')
except Exception as e:
    print(f'❌ Exception while sending email: {e}')
    sys.exit(1)
