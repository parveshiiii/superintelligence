import logging
from datetime import datetime

logging.basicConfig(filename='audit.log', level=logging.INFO)

class AuditLog:
    def __init__(self):
        self.log_entries = []

    def add_entry(self, user: str, action: str, details: str):
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        entry = f"{timestamp} - {user}: {action} - {details}"
        self.log_entries.append(entry)
        logging.info(entry)

    def get_entries(self):
        return self.log_entries