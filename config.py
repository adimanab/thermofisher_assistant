import os

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.pdf")
PERSIST_DIR = os.path.join(BASE_DIR, "pharma_db")
LOGO_PATH = os.path.join(BASE_DIR, "thermo_logo.jpg")

def validate_config():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    
    os.makedirs(PERSIST_DIR, exist_ok=True)
