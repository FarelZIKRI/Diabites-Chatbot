import os
import sys
from fastapi import FastAPI
from dotenv import load_dotenv

# Set project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Load env configurations
load_dotenv(os.path.join(BASE_DIR, '.env'))

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

from api.routes.chat import router as chat_router

# Inisialisasi FastAPI
app = FastAPI(
    title="DiaBites Chatbot API",
    description=(
        "Production REST API untuk DiaBites Hybrid Chatbot.\n\n"
        "Pipeline: IndoBERT Classification (Layer 1) -> Cosine Similarity (Lokal/Global) "
        "-> Groq Generative Fallback (Layer 2)"
    ),
    version="1.0.0"
)

# Registrasi Router
app.include_router(chat_router)

@app.get("/")
async def health_check():
    """Health check endpoint untuk verifikasi status server."""
    return {
        "status": "healthy",
        "app": "DiaBites Chatbot API",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "model": "IndoBERT + Cosine Similarity Hybrid",
        "fallback_model": GROQ_MODEL
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Menjalankan server DiaBites API di http://{HOST}:{PORT}")
    uvicorn.run("api.main:app", host=HOST, port=PORT, reload=True)
