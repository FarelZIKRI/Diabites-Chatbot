import os
import pickle
import numpy as np
import gc

# 1. Set environment variables to limit TensorFlow threading and memory overhead BEFORE importing it
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# 2. Now import TensorFlow and Keras first so transformers can detect the backend
import tensorflow as tf
import tf_keras as keras

# Explicitly configure TensorFlow C++ and Keras session limits
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
keras.backend.clear_session()
gc.collect()

# 3. Import transformers and load tokenizer
print("[..] Loading IndoBERT Tokenizer early for memory safety...")
from transformers import BertTokenizer, TFBertModel
try:
    from transformers import BertTokenizerFast
    GLOBAL_TOKENIZER = BertTokenizerFast.from_pretrained("indobenchmark/indobert-base-p2")
    print("[OK] BertTokenizerFast loaded successfully.")
except Exception:
    GLOBAL_TOKENIZER = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p2")
    print("[OK] Standard BertTokenizer loaded.")

# Trigger GC to clean up any temporary vocabulary parsing objects
gc.collect()

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENCODER_PATH = os.path.join(BASE_DIR, "data", "processed", "label_encoder.pkl")
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "model", "saved_model")
MODEL_WEIGHTS_PATH = os.path.join(SAVED_MODEL_DIR, "indobert_chatbot_weights.h5")

def build_head_model(num_classes=41):
    """Head model yang menerima embedding 768-dim dan output prediksi kelas."""
    input_emb = keras.layers.Input(shape=(768,), name="input_embeddings")
    dropout1 = keras.layers.Dropout(0.1)(input_emb)
    dense1 = keras.layers.Dense(512, activation='relu', name="dense_classification_1")(dropout1)
    dropout2 = keras.layers.Dropout(0.1)(dense1)
    dense2 = keras.layers.Dense(256, activation='relu', name="dense_classification_2")(dropout2)
    dropout3 = keras.layers.Dropout(0.1)(dense2)
    outputs = keras.layers.Dense(num_classes, activation='softmax', name="outputs")(dropout3)
    return keras.Model(inputs=input_emb, outputs=outputs)


class IntentClassifier:
    """
    Inference pipeline yang hemat memori:
    1. Load IndoBERT sebagai standalone (tanpa di-wrap model penuh)
    2. Load head model (Dense layers) terpisah dari weights
    3. Saat predict: tokenize -> BERT forward (1 sample) -> head predict
    """
    def __init__(self, encoder_path=ENCODER_PATH, weights_path=MODEL_WEIGHTS_PATH, max_len=32):
        self.max_len = max_len
        self.tokenizer = GLOBAL_TOKENIZER

        # Load Label Encoder
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label Encoder tidak ditemukan di: {encoder_path}")
        with open(encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)
        self.num_classes = len(self.label_encoder.classes_)
        print(f"[OK] Label Encoder loaded ({self.num_classes} classes).")

        # Load BERT sebagai standalone layer
        print("[..] Loading IndoBERT model untuk embedding extraction...")
        self.bert_model = TFBertModel.from_pretrained("indobenchmark/indobert-base-p2")
        self.bert_model.trainable = False
        print("[OK] IndoBERT loaded.")

        # Build & load head model
        print("[..] Loading classification head weights...")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Weights tidak ditemukan di: {weights_path}. "
                "Silakan jalankan model/train.py terlebih dahulu."
            )
        self.head_model = build_head_model(num_classes=self.num_classes)
        self.head_model.load_weights(weights_path, skip_mismatch=True, by_name=True)
        self.head_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("[OK] Classification head loaded!")

    def predict(self, query):
        """
        Melakukan tokenisasi, prediksi intent, dan ekstraksi CLS embedding.
        Returns:
            dict: { 'intent': str, 'confidence': float, 'embedding': np.ndarray }
        """
        # Tokenize
        encoded = self.tokenizer.encode_plus(
            str(query), 
            add_special_tokens=True, 
            max_length=self.max_len,
            padding='max_length', 
            truncation=True,
            return_attention_mask=True, 
            return_tensors='tf'
        )
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # Extract pooler_output from BERT standalone (to match pattern_embeddings.npy used during training)
        bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = bert_output.pooler_output.numpy()  # shape (1, 768)
        
        # Predict via head model
        preds = self.head_model.predict(embedding, verbose=0)[0]
        idx = np.argmax(preds)
        intent = self.label_encoder.inverse_transform([idx])[0]
        confidence = float(preds[idx])
        
        return {
            'intent': intent,
            'confidence': confidence,
            'embedding': embedding[0]  # flatten to (768,)
        }


def main():
    """Mode interaktif untuk testing."""
    print("=" * 60)
    print("  DiaBites Chatbot - Mode Interaktif")
    print("=" * 60)
    
    classifier = IntentClassifier()
    
    print("\nKetik pertanyaan Anda (ketik 'quit' untuk keluar):\n")
    while True:
        try:
            query = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSampai jumpa!")
            break
            
        if not query:
            continue
        if query.lower() in ('quit', 'exit', 'q'):
            print("Sampai jumpa!")
            break
        
        result = classifier.predict(query)
        print(f"   Intent     : {result['intent']}")
        print(f"   Confidence : {result['confidence']*100:.2f}%")
        print()

if __name__ == "__main__":
    main()
