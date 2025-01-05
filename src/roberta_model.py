import tensorflow as tf
from transformers import TFRobertaForSequenceClassification, RobertaConfig, RobertaTokenizer
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class RoBERTaClassifier:
    def __init__(self, 
                 num_classes=6,
                 lstm_units=200,
                 num_epochs=3, 
                 dropout_rate=0.2, 
                 learning_rate=2e-5,
                 model_save_dir='weights/roberta'):
        """
        Initialize the RoBERTa model with hyperparameters.
        """
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model_save_dir = model_save_dir

        # Build the RoBERTa-based classification model
        self.model = self.build_model()
        # Load tokenizer for preprocessing
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def build_model(self):
        """
        Builds and compiles a TFRobertaForSequenceClassification model with an additional LSTM layer.
        """
        # Load the RoBERTa model
        roberta_model = TFRobertaForSequenceClassification.from_pretrained(
            "roberta-base", 
            num_labels=self.num_classes, 
            from_pt=True
        )

        # Define input tensors explicitly
        input_ids = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name="input_ids")
        attention_mask = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name="attention_mask")

        # Pass the inputs to the roberta layer
        roberta_outputs = roberta_model.roberta(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        last_hidden_state = roberta_outputs.last_hidden_state

        # Add an LSTM layer
        lstm_layer = LSTM(self.lstm_units, return_sequences=False)(last_hidden_state)

        # Add a Dropout layer
        dropout_layer = Dropout(self.dropout_rate)(lstm_layer)

        # Add a Dense output layer
        output_layer = Dense(self.num_classes, activation='softmax')(dropout_layer)

        # Combine into a functional model
        model = Model(inputs=[input_ids, attention_mask], outputs=output_layer)

        # Compile the model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer, 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Trains the RoBERTa classifier.
        """
        callbacks = kwargs.pop('callbacks', [])
        if X_val is not None and y_val is not None:
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            callbacks.append(early_stopping)
            history = self.model.fit(
                X_train,
                y_train,
                epochs=self.num_epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                batch_size=8,
                verbose=1,
                **kwargs
            )
        else:
            history = self.model.fit(
                X_train,
                y_train,
                epochs=self.num_epochs,
                batch_size=8,
                verbose=1,
                callbacks=callbacks,
                **kwargs
            )

        return history
