import torch

async def load_model(model_type):
    if model_type == "nlp":
        # Load NLP model
        model = torch.hub.load('pytorch/fairseq', '