import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Pathing relative to this file
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

DATASET_PATH = ROOT_DIR / "data" / "dataset.pdf"
PERSIST_DIR = ROOT_DIR / "pharma_db"
LOGO_PATH = ROOT_DIR / "assets" / "thermo_logo.jpg"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# Name of your index in Pinecone
PINECONE_INDEX_NAME = os.getenv("INDEX_NAME", "thermo-assistant")  