from dotenv import load_dotenv
import os
import pandas as pd

# Load .env from the project root (assuming you're in /src or a subfolder)
load_dotenv(dotenv_path=os.path.join(os.getcwd(), '..', '.env'))

# Get the relative path from .env
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_PATH = os.path.join(BASE_DIR, os.getenv("DATA_PATH"))