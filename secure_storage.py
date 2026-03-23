import os
from encryption import encrypt_data, decrypt_data

def save_encrypted(file_path, data):
    encrypted = encrypt_data(data)
    with open(file_path, "wb") as f:
        f.write(encrypted)

def load_encrypted(file_path):
    with open(file_path, "rb") as f:
        encrypted = f.read()
    return decrypt_data(encrypted)