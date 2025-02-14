import tensorflow as tf
from transformers import TFRobertaForSequenceClassification, TFRobertaModel, RobertaConfig
from google.colab import drive
import os
import shutil

class RoBERTaClassifier:
    def __init__(self, num_classes=6, num_epochs=3, dropout_rate=0.2, learning_rate=2e-5):
        # Ensure the model uses GPU if available
        self.set_gpu_configuration()
        
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.feature_extractor = self.build_feature_extractor()
        
        # Mount Google Drive and set the save path
        self.save_path = self.setup_drive()

    def set_gpu_configuration(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU: {gpus[0].name}")
            except RuntimeError as e:
                print(f"Failed to set GPU memory growth: {e}")
        else:
            print("No GPU found. Using CPU.")
    
    def setup_drive(self):
        import sys
        if 'google.colab' in sys.modules:
            from google.colab import drive
                drive.mount('/content/drive')

        save_dir = "/content/drive/My Drive/weights"
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def build_model(self):
        config = RobertaConfig.from_pretrained("roberta-base", num_labels=self.num_classes)
        roberta_model = TFRobertaForSequenceClassification.from_pretrained("roberta-base", config=config)

        input_ids = tf.keras.Input(shape=(512,), dtype=tf.int32, name="input_ids")
        attention_mask = tf.keras.Input(shape=(512,), dtype=tf.int32, name="attention_mask")
        outputs = roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        
        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs.logits)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
        return model

    def build_feature_extractor(self):
        roberta_model = TFRobertaModel.from_pretrained("roberta-base")
        input_ids = tf.keras.Input(shape=(512,), dtype=tf.int32, name="input_ids")
        attention_mask = tf.keras.Input(shape=(512,), dtype=tf.int32, name="attention_mask")
        outputs = roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        feature_extractor = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=cls_embedding)
        return feature_extractor

    def extract_features(self, x):
        return self.feature_extractor.predict(x)

    def fit(self, x, y, validation_data=None, batch_size=32, verbose=1, **kwargs):
        kwargs.pop("epochs", None)
        history = self.model.fit(
            x=x,
            y=y,
            validation_data=validation_data,
            epochs=self.num_epochs,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs,
        )
        
        # Save model weights after training
        self.save_model_weights()
        return history

    def save_model_weights(self):
        weights_path = os.path.join(self.save_path, "weights.h5")
        zip_path = os.path.join(self.save_path, "weights.zip")
        
        # Save the model weights
        self.model.save_weights(weights_path)
        print(f"Model weights saved to: {weights_path}")
        
        # Zip the weights file
        shutil.make_archive(weights_path.replace('.h5', ''), 'zip', self.save_path)
        print(f"Model weights zipped and saved to: {zip_path}")

# Example Usage
if __name__ == "__main__":
    classifier = RoBERTaClassifier()
    sample_input = {
        "input_ids": tf.random.uniform((1, 512), minval=0, maxval=100, dtype=tf.int32),
        "attention_mask": tf.ones((1, 512), dtype=tf.int32),
    }
    feature_vectors = classifier.extract_features(sample_input)
    print("Extracted feature vector shape:", feature_vectors.shape)
