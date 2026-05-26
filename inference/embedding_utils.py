import numpy as np
import pandas as pd

def calculate_cosine_similarity(vec1, matrix):
    """
    Menghitung cosine similarity antara satu vektor dan matriks vektor.
    Args:
        vec1 (np.ndarray): Vektor query, shape (D,)
        matrix (np.ndarray): Matriks pattern embeddings, shape (N, D)
    Returns:
        np.ndarray: Skor similarity untuk setiap baris, shape (N,)
    """
    norm_vec1 = np.linalg.norm(vec1)
    norm_matrix = np.linalg.norm(matrix, axis=1)
    return np.dot(matrix, vec1) / (norm_matrix * norm_vec1 + 1e-9)

def local_similarity_search(query_embedding, pattern_embeddings, df, intent_tag):
    """
    Pencarian cosine similarity lokal khusus dalam intent terklasifikasi.
    Args:
        query_embedding (np.ndarray): Vektor query, shape (D,)
        pattern_embeddings (np.ndarray): Matriks pattern embeddings seluruh dataset
        df (pd.DataFrame): Dataframe dataset
        intent_tag (str): Tag intent hasil klasifikasi
    Returns:
        dict: { 'response': str, 'similarity_score': float }
    """
    local_df = df[df['tag'] == intent_tag]
    if local_df.empty:
        # Fallback jika intent tidak ditemukan di dataset
        return {
            'response': "Maaf, saya tidak menemukan jawaban yang cocok dalam kategori tersebut.",
            'similarity_score': 0.0
        }
        
    local_indices = local_df.index.values
    local_embeddings = pattern_embeddings[local_indices]
    
    sims = calculate_cosine_similarity(query_embedding, local_embeddings)
    best_idx = np.argmax(sims)
    global_idx = local_indices[best_idx]
    
    return {
        'response': df.iloc[global_idx]['response'],
        'similarity_score': float(sims[best_idx])
    }

def global_similarity_search(query_embedding, pattern_embeddings, df, top_k=4):
    """
    Pencarian cosine similarity global di seluruh dataset untuk fallback RAG.
    Args:
        query_embedding (np.ndarray): Vektor query, shape (D,)
        pattern_embeddings (np.ndarray): Matriks pattern embeddings seluruh dataset
        df (pd.DataFrame): Dataframe dataset
        top_k (int): Jumlah konteks teratas yang diambil
    Returns:
        list of dicts: [{ 'pattern': str, 'response': str, 'score': float }]
    """
    sims = calculate_cosine_similarity(query_embedding, pattern_embeddings)
    top_indices = np.argsort(sims)[::-1][:top_k]
    
    contexts = []
    for idx in top_indices:
        contexts.append({
            'pattern': df.iloc[idx]['pattern'],
            'response': df.iloc[idx]['response'],
            'score': float(sims[idx])
        })
    return contexts
