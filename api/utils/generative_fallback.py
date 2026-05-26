import os
from groq import Groq
from dotenv import load_dotenv

# Load env
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, '.env'))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

def call_groq_api(query: str, contexts: list, api_key: str = None, model_name: str = None) -> str:
    """
    Memanggil Groq API untuk asisten AI (generative fallback) menggunakan model LLaMA 3
    dengan menyuntikkan 3-5 konteks terdekat hasil pencarian similarity global.
    """
    api_key = api_key or GROQ_API_KEY
    model_name = model_name or GROQ_MODEL

    # Cek validitas API key
    if not api_key or api_key == "your_groq_api_key_here":
        return (
            "Maaf, asisten AI (Groq fallback) belum dikonfigurasi dengan API Key yang valid. "
            "Silakan hubungi administrator atau atur GROQ_API_KEY Anda di file .env."
        )

    # Susun teks konteks
    context_str = ""
    for idx, ctx in enumerate(contexts):
        context_str += (
            f"{idx + 1}. Pola: {ctx['pattern']}\n"
            f"   Respon: {ctx['response']}\n\n"
        )

    system_prompt = (
        "Anda adalah DiaBites Assistant, asisten AI edukasi diabetes dan gizi yang ramah.\n"
        "Tugas Anda adalah menjawab pertanyaan pengguna secara ringkas, santun, dan harus didasarkan pada konteks di bawah ini.\n\n"
        "Konteks Informasi DiaBites:\n"
        f"{context_str}"
        "Aturan:\n"
        "1. Jawablah hanya menggunakan informasi yang ada di dalam konteks di atas. Jangan mengarang data medis gizi di luar konteks.\n"
        "2. Jika pertanyaan pengguna tidak dapat dijawab langsung dari konteks di atas, katakan dengan sopan bahwa informasi Anda terbatas untuk hal tersebut dan sarankan untuk berkonsultasi dengan dokter.\n"
        "3. Gunakan Bahasa Indonesia yang baik, santun, dan komunikatif."
    )

    try:
        # Inisialisasi official Groq client
        client = Groq(api_key=api_key)
        
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.2,
            max_tokens=512
        )
        
        return chat_completion.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error calling Groq API (SDK): {e}")
        # Fallback ke raw HTTP request jika SDK mengalami kendala
        import requests
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.2,
            "max_tokens": 512
        }
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"Groq API HTTP Error: {response.status_code} - {response.text}")
                return "Maaf, sistem asisten AI sedang mengalami kendala. Silakan coba sesaat lagi."
        except Exception as http_err:
            print(f"Error calling Groq API (HTTP Fallback): {http_err}")
            return "Maaf, gagal menghubungi server asisten AI. Periksa koneksi internet Anda."
