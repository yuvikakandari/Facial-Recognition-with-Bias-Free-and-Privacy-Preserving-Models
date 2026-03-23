from cryptography.fernet import Fernet
import os

KEY_FILE = "secret.key"

def load_or_create_key():
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
    return open(KEY_FILE, "rb").read()

key = load_or_create_key()
fernet = Fernet(key)

def encrypt_data(data):
    return fernet.encrypt(data)

def decrypt_data(data):
    return fernet.decrypt(data)