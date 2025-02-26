class EthicalConstraints:
    def __init__(self):
        self.constraints = {
            "no_harm": "The AI must not make changes that could cause harm to users or the environment.",
            "privacy": "The AI must protect user privacy and not expose sensitive information.",
            "fairness": "The AI must ensure fairness and avoid bias in its decision-making processes."
        }

    def check_constraints(self, action: str, details: str) -> bool:
        # Implement logic to check if the action and details comply with the constraints
        for constraint, description in self.constraints.items():
            if not self._check_constraint(constraint, action, details):
                return False
        return True

    def _check_constraint(self, constraint: str, action: str, details: str) -> bool:
        # Implement specific checks for each constraint
        if constraint == "no_harm":
            return "harm" not in details
        elif constraint == "privacy":
            return "sensitive" not in details
        elif constraint == "fairness":
            return "bias" not in details
        return True