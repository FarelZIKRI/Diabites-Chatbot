import os
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Tambahkan project root ke sys.path jika belum ada
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference.predict import IntentClassifier
from inference.embedding_utils import local_similarity_search, global_similarity_search

# Load configurations
load_dotenv(os.path.join(BASE_DIR, '.env'))

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))

# Paths
DATA_CSV_PATH = os.path.join(BASE_DIR, "data", "diabites_chatbot_dataset.csv")
ENCODER_PATH = os.path.join(BASE_DIR, "data", "processed", "label_encoder.pkl")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "processed", "pattern_embeddings.npy")
MODEL_KERAS_PATH = os.path.join(BASE_DIR, "model", "saved_model", "indobert_chatbot.keras")
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "model", "saved_model", "indobert_chatbot_weights.h5")

# Global variables for lazy loading
_df = None
_pattern_embeddings = None
_classifier = None

def get_resources():
    """Lazy load resources to keep startup fast and avoid loading errors if files don't exist yet."""
    global _df, _pattern_embeddings, _classifier
    
    if _df is None:
        if not os.path.exists(DATA_CSV_PATH):
            raise FileNotFoundError(f"Dataset tidak ditemukan di {DATA_CSV_PATH}")
        _df = pd.read_csv(DATA_CSV_PATH)
        # Clean columns
        _df['tag'] = _df['tag'].str.strip()
        _df['pattern'] = _df['pattern'].str.strip()
        _df['response'] = _df['response'].str.strip()

    if _pattern_embeddings is None:
        if not os.path.exists(EMBEDDINGS_PATH):
            raise FileNotFoundError(f"Pattern Embeddings tidak ditemukan di {EMBEDDINGS_PATH}")
        _pattern_embeddings = np.load(EMBEDDINGS_PATH)

    if _classifier is None:
        _classifier = IntentClassifier(
            encoder_path=ENCODER_PATH,
            weights_path=MODEL_WEIGHTS_PATH
        )
        
    return _df, _pattern_embeddings, _classifier

def get_chatbot_response(query: str) -> dict:
    """
    Memproses kueri pengguna menggunakan dual-layer pipeline:
    - Layer 1: Cosine Similarity Lokal (Confidence >= threshold)
    - Layer 2: Cosine Similarity Global + Groq Fallback RAG (Confidence < threshold)
    """
    query = query.strip()
    if not query:
        return {
            'response': "Pesan tidak boleh kosong.",
            'intent': "none",
            'confidence': 0.0,
            'source': 'error'
        }

    try:
        # Load resources lazily
        df, pattern_embeddings, classifier = get_resources()
        
        # Step 1: Prediksi intent & CLS embedding
        pred = classifier.predict(query)
        intent = pred['intent']
        confidence = pred['confidence']
        embedding = pred['embedding']

        # Step 2: Routing berdasarkan threshold confidence
        if confidence >= CONFIDENCE_THRESHOLD:
            # --- Layer 1: Cosine Similarity Lokal ---
            try:
                res = local_similarity_search(embedding, pattern_embeddings, df, intent)
                return {
                    'response': res['response'],
                    'intent': intent,
                    'confidence': confidence,
                    'source': 'classification_local'
                }
            except Exception as e:
                # Fallback ke respons pertama pada intent jika similarity lokal gagal
                fallback_res = df[df['tag'] == intent].iloc[0]['response']
                return {
                    'response': fallback_res,
                    'intent': intent,
                    'confidence': confidence,
                    'source': 'classification_local_fallback'
                }
        else:
            # --- Layer 2: Cosine Similarity Global + Groq Fallback RAG ---
            try:
                contexts = global_similarity_search(embedding, pattern_embeddings, df, top_k=4)
                
                # Import dynamic to avoid circular import and allow standalone testing of inference
                from api.utils.generative_fallback import call_groq_api
                
                response_text = call_groq_api(query, contexts)
                return {
                    'response': response_text,
                    'intent': intent,
                    'confidence': confidence,
                    'source': 'generative_fallback'
                }
            except Exception as e:
                print(f"Fallback RAG Error: {e}")
                return {
                    'response': "Maaf, saya belum memahami pertanyaan Anda secara mendalam. Silakan tanyakan hal lain seputar diabetes atau nutrisi.",
                    'intent': intent,
                    'confidence': confidence,
                    'source': 'error_fallback'
                }
    except Exception as e:
        return {
            'response': f"Gagal memproses kueri: {e}",
            'intent': "error",
            'confidence': 0.0,
            'source': 'error'
        }
