import hashlib

class PasswordManager:
    def __init__(self):
        # Store the hashed password
        self.hashed_password = self.hash_password("ab12qweasdzxc")

    def hash_password(self, password: str) -> str:
        # Hash the password using SHA-256
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, password: str) -> bool:
        # Verify the input password against the stored hashed password
        return self.hash_password(password) == self.hashed_password