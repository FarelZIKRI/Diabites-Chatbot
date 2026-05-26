import os
import pickle
import numpy as np
import tensorflow as tf
import tf_keras as keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.npz")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "processed", "pattern_embeddings.npy")
ENCODER_PATH = os.path.join(BASE_DIR, "data", "processed", "label_encoder.pkl")
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "model", "saved_model")
MODEL_WEIGHTS_PATH = os.path.join(SAVED_MODEL_DIR, "indobert_chatbot_weights.h5")

def build_head_model(num_classes=41):
    """Membangun model klasifikasi mandiri di atas raw CLS embeddings (768 dimensi).
    Arsitektur harus identik dengan yang ada di train.py."""
    input_emb = keras.layers.Input(shape=(768,), name="input_embeddings")
    dropout1 = keras.layers.Dropout(0.1)(input_emb)
    dense1 = keras.layers.Dense(
        512, 
        activation='relu',
        name="dense_classification_1"
    )(dropout1)
    dropout2 = keras.layers.Dropout(0.1)(dense1)
    dense2 = keras.layers.Dense(
        256, 
        activation='relu',
        name="dense_classification_2"
    )(dropout2)
    dropout3 = keras.layers.Dropout(0.1)(dense2)
    outputs = keras.layers.Dense(num_classes, activation='softmax', name="outputs")(dropout3)

    model = keras.Model(inputs=input_emb, outputs=outputs)
    return model

def main():
    print("=" * 60)
    print("  DiaBites Chatbot - Evaluasi Model (Ringan, Tanpa Load BERT)")
    print("=" * 60)

    # 1. Load Pre-computed Embeddings & Labels
    if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(NPZ_PATH):
        raise FileNotFoundError("File embeddings (.npy) atau data (.npz) tidak ditemukan di data/processed/.")
    
    embeddings = np.load(EMBEDDINGS_PATH)
    data = np.load(NPZ_PATH)
    labels = data['labels']
    print(f"[OK] Pre-computed embeddings loaded. Shape: {embeddings.shape}")

    # 2. Load Label Encoder
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"File label encoder tidak ditemukan di: {ENCODER_PATH}")
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    num_classes = len(le.classes_)
    print(f"[OK] Label Encoder loaded ({num_classes} kelas)")

    # 3. Stratified Train-Validation Split (harus sama persis dengan train.py)
    _, val_embs, _, val_labels = train_test_split(
        embeddings, labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )
    print(f"[OK] Validasi data loaded. Jumlah: {len(val_labels)} sampel.")

    # 4. Build Head Model & Load Trained Weights
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights model tidak ditemukan di: {MODEL_WEIGHTS_PATH}. Jalankan model/train.py terlebih dahulu.")
        
    print("[..] Membangun head model dan memuat weights...")
    head_model = build_head_model(num_classes=num_classes)
    
    # Load only the Dense layer weights from the saved full model weights using legacy by_name loading
    head_model.load_weights(MODEL_WEIGHTS_PATH, skip_mismatch=True, by_name=True)
    
    head_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("[OK] Head model weights berhasil dimuat!")

    # 5. Predict pada validation set
    print("[..] Memprediksi data validasi...")
    preds = head_model.predict(val_embs, batch_size=32, verbose=1)
    pred_labels = np.argmax(preds, axis=1)

    # 6. Metrics Calculation
    acc = accuracy_score(val_labels, pred_labels)
    print("\n" + "-" * 40)
    print(f" AKURASI MODEL: {acc * 100:.2f}%")
    print("-" * 40)

    # Target Quest check
    target_accuracy = 0.90
    if acc >= target_accuracy:
        print(f"[SUCCESS] Akurasi Terpenuhi! Akurasi ({acc*100:.2f}%) >= Target ({target_accuracy*100:.2f}%)")
    else:
        print(f"[FAILED] AkurasiBelum Terpenuhi! Akurasi ({acc*100:.2f}%) < Target ({target_accuracy*100:.2f}%)")

    # 7. Evaluasi pada SELURUH dataset (untuk confidence report)
    print("\n[..] Memprediksi seluruh dataset...")
    all_preds = head_model.predict(embeddings, batch_size=32, verbose=1)
    all_pred_labels = np.argmax(all_preds, axis=1)
    all_acc = accuracy_score(labels, all_pred_labels)
    print(f" AKURASI SELURUH DATASET: {all_acc * 100:.2f}%")

    print("\nClassification Report (Validation Set):")
    report = classification_report(
        val_labels, 
        pred_labels, 
        target_names=le.classes_, 
        digits=4
    )
    print(report)
    print("=" * 60)

if __name__ == "__main__":
    main()
