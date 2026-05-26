import os
import pickle
import numpy as np
import tensorflow as tf
import tf_keras
tf.keras = tf_keras
from sklearn.model_selection import train_test_split
from transformers import TFBertModel
from custom_callbacks import ConfidenceLogger

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.npz")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "processed", "pattern_embeddings.npy")
ENCODER_PATH = os.path.join(BASE_DIR, "data", "processed", "label_encoder.pkl")
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "model", "saved_model")
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

MODEL_KERAS_PATH = os.path.join(SAVED_MODEL_DIR, "indobert_chatbot.keras")
MODEL_WEIGHTS_PATH = os.path.join(SAVED_MODEL_DIR, "indobert_chatbot_weights.h5")
TENSORBOARD_LOG_DIR = os.path.join(BASE_DIR, "logs", "tensorboard")
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

def build_functional_model(max_len=32, num_classes=41, model_name='indobenchmark/indobert-base-p2'):
    """Membangun arsitektur Keras Functional API dengan IndoBERT menggunakan raw CLS token slicing."""
    input_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    # Load IndoBERT base model
    bert_layer = TFBertModel.from_pretrained(model_name)
    bert_layer.trainable = False  # Freeze BERT layers

    bert_outputs = bert_layer(input_ids=input_ids, attention_mask=attention_mask)
    # Extract raw CLS token embedding from last_hidden_state (index 0)
    pooled_output = bert_outputs.last_hidden_state[:, 0, :]

    # Classification Head
    dropout1 = tf.keras.layers.Dropout(0.1)(pooled_output)
    dense1 = tf.keras.layers.Dense(
        512, 
        activation='relu',
        name="dense_classification_1"
    )(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.1)(dense1)
    dense2 = tf.keras.layers.Dense(
        256, 
        activation='relu',
        name="dense_classification_2"
    )(dropout2)
    dropout3 = tf.keras.layers.Dropout(0.1)(dense2)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name="outputs")(dropout3)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)
    return model

def build_head_model(num_classes=41):
    """Membangun model klasifikasi mandiri di atas raw CLS embeddings (768 dimensi)."""
    input_emb = tf.keras.layers.Input(shape=(768,), name="input_embeddings")
    dropout1 = tf.keras.layers.Dropout(0.1)(input_emb)
    dense1 = tf.keras.layers.Dense(
        512, 
        activation='relu',
        name="dense_classification_1"
    )(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.1)(dense1)
    dense2 = tf.keras.layers.Dense(
        256, 
        activation='relu',
        name="dense_classification_2"
    )(dropout2)
    dropout3 = tf.keras.layers.Dropout(0.1)(dense2)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name="outputs")(dropout3)

    model = tf.keras.Model(inputs=input_emb, outputs=outputs)
    return model

def main():
    print("=" * 60)
    print("  DiaBites Chatbot - Memulai Proses Training Teroptimasi (Detik)")
    print("=" * 60)

    # 1. Load Pre-computed Embeddings & Targets
    if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(NPZ_PATH):
        raise FileNotFoundError("File embeddings (.npy) atau data (.npz) tidak ditemukan di data/processed/.")
    
    embeddings = np.load(EMBEDDINGS_PATH)
    data = np.load(NPZ_PATH)
    labels = data['labels']
    print(f"[OK] Pre-computed raw CLS embeddings loaded. Shape: {embeddings.shape}")

    # 2. Load Label Encoder
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"File label encoder tidak ditemukan di: {ENCODER_PATH}")
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    num_classes = len(le.classes_)
    print(f"[OK] Label Encoder loaded ({num_classes} kelas)")

    # 3. Stratified Train-Validation Split (80% Train, 20% Val)
    train_embs, val_embs, train_labels, val_labels = train_test_split(
        embeddings, labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )
    print(f"[OK] Split data: Train={len(train_labels)}, Validation={len(val_labels)}")

    # 4. Build Standalone Head Model
    print("[..] Membangun model klasifikasi Dense head...")
    head_model = build_head_model(num_classes=num_classes)
    head_model.summary()

    # Compile Head Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    head_model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("[OK] Head model compiled successfully")

    # 5. Callbacks
    # Custom confidence logger for validation embeddings (no emojis to prevent Windows console encoding crashes)
    conf_logger = ConfidenceLogger(val_data=(val_embs, val_labels))
    
    # TensorBoard logs (histogram_freq=0 to prevent OOM)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=TENSORBOARD_LOG_DIR, 
        histogram_freq=0
    )
    
    callbacks = [conf_logger, tensorboard_callback]

    # 6. Fit Head Model (Sangat cepat - hanya dalam hitungan detik!)
    print("[..] Melatih classification head di atas seluruh embeddings...")
    head_model.fit(
        x=embeddings,
        y=labels,
        validation_data=(val_embs, val_labels),
        epochs=150,  # 150 epoch untuk menjamin konvergensi karena sangat cepat
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    print("[OK] Training classification head selesai!")

    # Save head weights FIRST!
    print(f"[..] Menyimpan weights ke {MODEL_WEIGHTS_PATH}...")
    head_model.save_weights(MODEL_WEIGHTS_PATH)
    print(f"[OK] Weights tersimpan!")

    # 7. Stitch Head Model Weights into Full Model with BERT (Optional / Wrapped in Try-Except to handle RAM limits)
    try:
        print("\n[..] Membebaskan memori sebelum memuat model penuh...")
        tf.keras.backend.clear_session()

        print("[..] Membangun Model Penuh Keras Functional API dengan IndoBERT...")
        full_model = build_functional_model(max_len=32, num_classes=num_classes)
        
        print("[..] Mentransfer weights hasil training ke model penuh...")
        # Find dense layers in both models
        dense_full = [l for l in full_model.layers if isinstance(l, tf.keras.layers.Dense)]
        
        # Copy weights
        for f_layer in dense_full:
            h_layer = head_model.get_layer(f_layer.name)
            print(f"Mentransfer weights untuk layer: {f_layer.name}")
            f_layer.set_weights(h_layer.get_weights())
            
        print("[OK] Transfer weights berhasil!")

        # Compile full model so it's ready for evaluation/inference
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Save stitched model
        print(f"\n[..] Menyimpan model penuh ter-stitch ke {MODEL_KERAS_PATH}...")
        full_model.save(MODEL_KERAS_PATH)
        print(f"[OK] Model penuh tersimpan!")
    except Exception as e:
        print(f"\n[WARNING] Tidak dapat melakukan stitch model penuh karena keterbatasan RAM/Memory: {e}")
        print("[INFO] Model weights head model telah berhasil disimpan dan dapat digunakan untuk evaluasi & inferensi.")

    print("=" * 60)

if __name__ == "__main__":
    main()
