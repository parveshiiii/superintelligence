class PermissionManager:
    def __init__(self):
        # Initialize with default permissions
        self.permissions = {}

    def set_permission(self, file_path: str, user: str, permission: str):
        if file_path not in self.permissions:
            self.permissions[file_path] = {}
        self.permissions[file_path][user] = permission

    def is_edit_allowed(self, file_path: str, user: str) -> bool:
        return self.permissions.get(file_path, {}).get(user, '') == 'edit'

    def is_create_allowed(self, file_path: str, user: str) -> bool:
        return self.permissions.get(file_path, {}).get(user, '') == 'create'