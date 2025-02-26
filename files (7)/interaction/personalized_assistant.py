import torch
import torch.nn as nn

class PersonalizedAI(nn.Module):
    def __init__(self, user_preferences):
        super(PersonalizedAI, self).__init__()
        self.user_preferences = user_preferences
        self.fc = nn.Linear(len(user_preferences), 10)

    def forward(self, x):
        preferences_tensor = torch.tensor(self.user_preferences, dtype=torch.float32)
        x = torch.cat((x, preferences_tensor), dim=1)
        x = self.fc(x)
        return x