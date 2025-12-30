import os

# Try to load .env if python-dotenv is available (same behavior as email_sender)
try:
	from dotenv import load_dotenv
	ok = load_dotenv()
	print('dotenv.load_dotenv() ->', ok)
except Exception as e:
	print('dotenv import/load failed:', e)

import os as _os
print('cwd =', _os.getcwd())
print('.env exists =', _os.path.exists('.env'))

print('ALERT_SENDER_EMAIL=', os.getenv('ALERT_SENDER_EMAIL'))
print('ALERT_EMAIL_PASSWORD=', '***' if os.getenv('ALERT_EMAIL_PASSWORD') else '(not set)')
print('ALERT_TO_EMAIL=', os.getenv('ALERT_TO_EMAIL'))
