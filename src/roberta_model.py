import tensorflow as tf
from transformers import TFRobertaForSequenceClassification, RobertaConfig

class RoBERTaClassifier:
    def __init__(self, num_classes=6, lstm_units=200, num_epochs=3, dropout_rate=0.2, learning_rate=2e-5):
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self.build_model()

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

    def fit(self, train_data, y_train, X_val=None, y_val=None):
        callbacks = []

        if X_val is not None and y_val is not None:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
            callbacks.append(early_stopping)
            history = self.model.fit(
                train_data,
                y_train,
                validation_data=(X_val, y_val),
                epochs=self.num_epochs,
                batch_size=32,
                callbacks=callbacks,
                verbose=1,
            )
        else:
            history = self.model.fit(
                train_data,
                y_train,
                epochs=self.num_epochs,
                batch_size=32,
                callbacks=callbacks,
                verbose=1,
            )

        return history
