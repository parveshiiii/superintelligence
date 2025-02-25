import random
import datetime
import time
import logging
import asyncio
import torch
import faiss
import pickle
import numpy as np
import matplotlib.pyplot as plt
import requests
import feedparser  # For parsing RSS feeds
from sentence_transformers import SentenceTransformer
from model_loader import load_model
from git import Repo

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
EM