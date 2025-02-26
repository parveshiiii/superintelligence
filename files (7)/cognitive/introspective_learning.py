from datetime import datetime

class IntrospectiveLearning:
    def __init__(self):
        self.experience_log = []

    def log_experience(self, experience):
        timestamp = datetime.utcnow()
        self.experience_log.append((timestamp, experience))

    def introspect(self):
        insights = []
        for timestamp, experience in self.experience_log:
            insights.append(f"At {timestamp}, experience was: {experience}")
        return insights