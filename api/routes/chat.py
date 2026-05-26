import os
import sys
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Project Root setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference.response_handler import get_chatbot_response

router = APIRouter()

# Schema Pydantic
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float
    source: str

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Endpoint utama asisten chatbot DiaBites.
    Menerima pesan kueri dari pengguna dan memprosesnya dengan dual-layer pipeline:
    IndoBERT (Layer 1) -> Cosine Similarity (Lokal/Global) -> Groq RAG Fallback (Layer 2).
    """
    query = req.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Pesan tidak boleh kosong.")

    res = get_chatbot_response(query)
    if res['source'] == 'error':
        raise HTTPException(status_code=500, detail=res['response'])

    return ChatResponse(**res)
