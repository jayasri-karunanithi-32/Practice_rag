from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Batch processing and embedding settings
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    INGESTION_BATCH_SIZE = int(os.getenv("INGESTION_BATCH_SIZE", "5"))
    MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))
    RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "0.0"))
    
settings = Settings()
