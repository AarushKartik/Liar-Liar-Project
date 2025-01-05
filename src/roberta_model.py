import tensorflow as tf
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
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
        class CustomRoBERTaModel(Model):
            def __init__(self, num_classes, lstm_units, dropout_rate, learning_rate):
                super(CustomRoBERTaModel, self).__init__()
                self.roberta = TFRobertaForSequenceClassification.from_pretrained(
                    "roberta-base",
                    num_labels=num_classes,
                    from_pt=True
                ).roberta
                self.lstm = LSTM(lstm_units, return_sequences=False)
                self.dropout = Dropout(dropout_rate)
                self.classifier = Dense(num_classes, activation="softmax")

            def call(self, inputs):
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = roberta_output.last_hidden_state
                x = self.lstm(last_hidden_state)
                x = self.dropout(x)
                output = self.classifier(x)
                return output

        # Initialize the custom model
        model = CustomRoBERTaModel(self.num_classes, self.lstm_units, self.dropout_rate, self.learning_rate)

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"]
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
