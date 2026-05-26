import tensorflow as tf
import numpy as np

class ConfidenceLogger(tf.keras.callbacks.Callback):
    """
    Custom Keras callback that computes and logs the average prediction confidence
    (the maximum probability output of the softmax layer) on training and/or validation data.
    """
    def __init__(self, val_data=None):
        super().__init__()
        # val_data should be a tuple of (inputs, targets) or a tf.data.Dataset
        self.val_data = val_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Calculate training confidence (if y_pred/softmax outputs are accessible)
        # However, to be extremely clean and robust, we can compute confidence 
        # on the validation data if it's passed to the callback.
        if self.val_data is not None:
            if isinstance(self.val_data, tuple):
                x_val, y_val = self.val_data
            else:
                # Assume tf.data.Dataset
                x_val, y_val = next(iter(self.val_data.batch(len(self.val_data))))
                
            preds = self.model.predict(x_val, verbose=0)
            avg_confidence = float(np.mean(np.max(preds, axis=1)))
            logs['val_avg_confidence'] = avg_confidence
            print(f"\n - epoch {epoch + 1} - val_avg_confidence: {avg_confidence:.4f}")
        else:
            # If validation data isn't passed, we can print a message
            pass
