import shutil
import os
from datetime import datetime

class RecoveryMechanism:
    def __init__(self, backup_dir: str):
        self.backup_dir = backup_dir

    def create_backup(self, project_dir: str):
        backup_path = os.path.join(self.backup_dir, datetime.utcnow().strftime('%Y%m%d%H%M%S'))
        shutil.copytree(project_dir, backup_path)
        return backup_path

    def restore_backup(self, backup_path: str, project_dir: str):
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
        shutil.copytree(backup_path, project_dir)