from cryptography.fernet import Fernet

# Generate a Fernet key
key = Fernet.generate_key()
print(key.decode())  # Decode to string for use in secrets.toml