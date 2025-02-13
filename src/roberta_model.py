import tensorflow as tf
from transformers import TFRobertaForSequenceClassification, TFRobertaModel, RobertaConfig

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

    def set_gpu_configuration(self):
        # Check if a GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Set memory growth to avoid OOM errors
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU: {gpus[0].name}")
            except RuntimeError as e:
                print(f"Failed to set GPU memory growth: {e}")
        else:
            print("No GPU found. Using CPU.")

    def build_model(self):
        # Load a RoBERTa configuration with the desired number of labels
        config = RobertaConfig.from_pretrained("roberta-base", num_labels=self.num_classes)

        # Initialize the TFRobertaForSequenceClassification model
        roberta_model = TFRobertaForSequenceClassification.from_pretrained("roberta-base", config=config)

        # Define the inputs
        input_ids = tf.keras.Input(shape=(512,), dtype=tf.int32, name="input_ids")
        attention_mask = tf.keras.Input(shape=(512,), dtype=tf.int32, name="attention_mask")

        # Forward pass through the RoBERTa model
        outputs = roberta_model(input_ids=input_ids, attention_mask=attention_mask)

        # Create the final model
        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs.logits)

        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model

    def build_feature_extractor(self):
        # Load the base RoBERTa model without classification head
        roberta_model = TFRobertaModel.from_pretrained("roberta-base")

        # Define inputs
        input_ids = tf.keras.Input(shape=(512,), dtype=tf.int32, name="input_ids")
        attention_mask = tf.keras.Input(shape=(512,), dtype=tf.int32, name="attention_mask")

        # Extract hidden states
        outputs = roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_dim)

        # Apply pooling to get a fixed-size feature vector (CLS token)
        cls_embedding = hidden_states[:, 0, :]  # CLS token representation

        # Create the feature extraction model
        feature_extractor = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=cls_embedding)

        return feature_extractor

    def extract_features(self, x):
        """
        Extract feature vectors from the model.

        Args:
            x: Input dictionary with keys 'input_ids' and 'attention_mask'.

        Returns:
            Feature vectors (numpy array) of shape (batch_size, hidden_dim).
        """
        return self.feature_extractor.predict(x)

    def fit(self, x, y, validation_data=None, batch_size=32, verbose=1, **kwargs):
        """
        Fit the model to the training data.
        
        Args:
            x: Input data (dictionary of 'input_ids' and 'attention_mask').
            y: Labels for training.
            validation_data: Tuple (X_val, y_val) for validation.
            **kwargs: Additional arguments for the model's fit method.
        """
        kwargs.pop("epochs", None)
        if validation_data is not None:
            X_val, y_val = validation_data
            history = self.model.fit(
                x=x,
                y=y,
                validation_data=(X_val, y_val),
                epochs=self.num_epochs,
                batch_size=batch_size,
                verbose=verbose,
                **kwargs,
            )
        else:
            history = self.model.fit(
                x=x,
                y=y,
                epochs=self.num_epochs,
                batch_size=batch_size,
                verbose=verbose,
                **kwargs,
            )

        return history

# Example Usage
if __name__ == "__main__":
    classifier = RoBERTaClassifier()
    # Example input (replace with real tokenized input)
    sample_input = {
        "input_ids": tf.random.uniform((1, 512), minval=0, maxval=100, dtype=tf.int32),
        "attention_mask": tf.ones((1, 512), dtype=tf.int32),
    }
    feature_vectors = classifier.extract_features(sample_input)
    print("Extracted feature vector shape:", feature_vectors.shape)
