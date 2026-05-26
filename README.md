# DiaBites Chatbot AI

Fitur Chatbot AI untuk aplikasi **DiaBites** — PWA yang membantu penyandang diabetes mengevaluasi kandungan gizi produk kemasan. Chatbot ini menggunakan pipeline cerdas dua lapis (Dual-Layer Hybrid Pipeline) yang dioptimalkan khusus untuk efisiensi CPU dan memori rendah (RAM-friendly):

- **Layer 1 — IndoBERT Classifier + Cosine Similarity Lokal (Local Search):** Klasifikasi intent dari kueri pengguna menggunakan model Dense Classifier mandiri di atas representasi IndoBERT (`pooler_output`). Jika confidence skor klasifikasi intent $\ge$ `CONFIDENCE_THRESHOLD` (default: `0.85`), kecocokan dicari menggunakan _Cosine Similarity_ lokal pada intent terpilih untuk mengambil jawaban paling relevan dari dataset.
- **Layer 2 — Cosine Similarity Global + Groq API LLaMA 3 (Generative Fallback RAG):** Aktif secara otomatis jika confidence skor intent kurang dari threshold. Model melakukan pencarian kemiripan kosinus secara global di seluruh dataset untuk mengekstrak 4 konteks terdekat, lalu mengirimkannya sebagai basis pengetahuan (RAG) ke Groq API (`llama-3.1-8b-instant`) untuk merumuskan respons yang dinamis, akurat, dan natural.

---

## Struktur Direktori Proyek

```
diabites-chatbot/
├── data/
│   ├── diabites_chatbot_dataset.csv     # Dataset kueri intent & respons DiaBites (1.310 baris, 41 kelas)
│   └── processed/
│       ├── label_encoder.pkl            # Objek encoder kategori intent (41 kelas)
│       ├── processed_data.npz           # Tokenized input_ids & attention_mask hasil preprocessing
│       └── pattern_embeddings.npy       # Precomputed pooler_output embeddings (1310x768)
├── model/
│   ├── train.py                         # Skrip latih mandiri untuk Dense classification head
│   ├── custom_callbacks.py              # Logger kustom untuk memantau confidence score saat training
│   ├── evaluate.py                      # Skrip pengujian akurasi model & laporan klasifikasi (F1-score)
│   └── saved_model/
│       ├── indobert_chatbot_weights.h5  # Bobot latih model Dense head Keras 2 kompatibel (Ringan: ~2.1 MB)
│       └── indobert_chatbot.keras       # Model penuh ter-stitch (Stitched) dengan IndoBERT (Opsional: ~500 MB)
├── inference/
│   ├── predict.py                       # Pipeline inferensi klasifikasi intent hemat memori & OOM-free
│   ├── embedding_utils.py               # Fungsi utilitas Cosine Similarity lokal & global
│   └── response_handler.py              # Handler pipeline hybrid Dual-Layer (Layer 1 -> Layer 2 RAG)
├── api/
│   ├── main.py                          # REST API Entrypoint utama menggunakan FastAPI
│   ├── routes/
│   │   └── chat.py                      # Router endpoint API /chat
│   └── utils/
│       └── generative_fallback.py       # Handler komunikasi eksternal ke Groq API RAG Fallback
├── notebooks/
│   ├── 01_EDA_preprocessing.ipynb       # Notebook pembersihan data, tokenisasi, & ekstraksi embeddings
│   └── 02_training_evaluation.ipynb     # Notebook alternatif training & evaluasi interaktif
├── logs/
│   ├── tensorboard/                     # Log training Tensorboard
│   └── training_curves.png              # Grafik performa akurasi & loss training
├── .env.example                         # Contoh file konfigurasi environment
├── .gitignore                           # Konfigurasi pengecualian unggahan Git (termasuk .env & weights besar)
├── requirements.txt                     # Daftar pustaka dependencies proyek
└── README.md                            # Dokumentasi teknis proyek
```

---

## Optimalisasi Memori Lokal (OOM & MemoryError Prevention)

Untuk memastikan kelancaran eksekusi pada komputer CPU lokal atau sistem dengan RAM terbatas, proyek ini menerapkan strategi **Memory-Defensive**:

1. **Capping Threads**: Pembatasan thread internal TensorFlow CPU diatur ketat menjadi `1` thread menggunakan variabel environment (`OMP_NUM_THREADS="1"`, dll.) untuk mencegah membengkaknya memori akibat thread-pools bawaan.
2. **Early Tokenizer Loading**: Tokenizer dimuat paling awal saat kondisi memori RAM bersih sebelum TensorFlow dimpor untuk mencegah kegagalan alokasi memori kamus kosakata (_vocabulary parsing_).
3. **Fast Tokenizer Fallback**: Mendukung pemuatan otomatis `BertTokenizerFast` berbasis Rust yang sangat hemat memori dan cepat.
4. **Lightweight Load Weights**: Modul evaluasi dan inferensi hanya memuat bobot Dense klasifikasi (`indobert_chatbot_weights.h5` - 2.1 MB) alih-alih model penuh (500 MB), menghemat penggunaan RAM lebih dari $80\%$.

---

## Urutan Menjalankan Pipeline (Lokal)

### **Langkah 1: Eksplorasi Data & Pra-pemrosesan**

Jalankan file pertama untuk memproses dataset mentah menjadi representasi token.

- **File**: `notebooks/01_EDA_preprocessing.ipynb`
- **Hasil**: Meng-encode intent (`label_encoder.pkl`), menghasilkan token data (`processed_data.npz`), dan mengekstrak `pooler_output` embedding kalimat dari IndoBERT (`pattern_embeddings.npy`).

### **Langkah 2: Pelatihan Model (Training)**

Latih Dense Head Classifier di atas embedding yang telah di-pra-komputasi.

- **File**: `model/train.py` (atau alternatif interaktif di `notebooks/02_training_evaluation.ipynb`)
- **Hasil**: Melatih klasifikasi intent dengan cepat (hitungan detik) dan menyimpan bobot teruji (`indobert_chatbot_weights.h5`).

### **Langkah 3: Evaluasi Akurasi Model**

Uji performa model di atas data validasi secara objektif.

- **File**: `model/evaluate.py`
- **Hasil**: Menampilkan akurasi akhir serta tabel laporan klasifikasi komprehensif (_F1-Score_, _Precision_, _Recall_).

### **Langkah 4: Menjalankan Server API FastAPI**

Nyalakan server lokal untuk menerima chat.

- **File**: `api/main.py`
- **Hasil**: API Server siap melayani kueri di `http://127.0.0.1:8000/`.

## Cara Testing API

Setelah server API aktif, gunakan terminal/Command Prompt baru untuk menguji chat:

```bash
# Uji Health Check Endpoint
curl http://127.0.0.1:8000/

# Uji Chat Endpoint (Contoh kueri lokal - confidence tinggi)
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Apakah edit manual mempengaruhi akurasi rekomendasi"}'

## Tech Stack Utama

- **Model Klasifikasi:** TensorFlow 2.x + tf_keras (Keras Functional API)
- **Natural Language Processing:** IndoBERT (`indobenchmark/indobert-base-p2`) via HuggingFace Transformers
- **Web Framework API:** FastAPI + Uvicorn
- **Generative AI Fallback:** Groq API SDK (Model: `llama-3.1-8b-instant`)
- **Pencarian Kemiripan:** Cosine Similarity Lokal/Global (NumPy)

```
