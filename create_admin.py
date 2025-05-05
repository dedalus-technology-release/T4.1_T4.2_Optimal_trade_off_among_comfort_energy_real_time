import os
from dotenv import load_dotenv
from models import create_user, get_user_by_username
from database import init_db

load_dotenv()

ADMIN_USERNAME = os.getenv('ADMIN_USERNAME')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')

if not ADMIN_USERNAME or not ADMIN_PASSWORD:
    raise ValueError("ADMIN_USERNAME and ADMIN_PASSWORD must be set as environment variables.")

if get_user_by_username(ADMIN_USERNAME):
    print(f"[!] Admin user '{ADMIN_USERNAME}' already exists. Skipping creation.")
else:
    init_db()
    create_user(ADMIN_USERNAME, ADMIN_PASSWORD, is_admin=True)
    print(f"[+] Admin user '{ADMIN_USERNAME}' created successfully.")