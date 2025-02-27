import os
import shutil
import numpy as np
import tensorflow as tf
from transformers import TFRobertaForSequenceClassification, TFRobertaModel, RobertaConfig

class RoBERTaClassifier:
    def __init__(self, num_classes=6, num_epochs=3, dropout_rate=0.1298723500192933, learning_rate=1.85e-05, batch_size=16, model_save_dir="weights"):
        # Ensure the model uses GPU if available
        self.set_gpu_configuration()
        
        self.num_classes = num_classe
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model_save_dir = os.path.abspath(model_save_dir)  # Directory to save model weights

        # Ensure directory exists
        os.makedirs(self.model_save_dir, exist_ok=True)

        self.model = self.build_model()

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

    def build_model(self):
        config = RobertaConfig.from_pretrained("roberta-base", num_labels=self.num_classes)
        roberta_model = TFRobertaForSequenceClassification.from_pretrained("roberta-base", config=config)

        input_ids = tf.keras.Input(shape=(512,), dtype=tf.int32, name="input_ids")
        attention_mask = tf.keras.Input(shape=(512,), dtype=tf.int32, name="attention_mask")

        outputs = roberta_model(input_ids=input_ids, attention_mask=attention_mask)

        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs.logits)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, jit_compile=False)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
        return model

    def fit(self, x, y, validation_data=None, batch_size=32, verbose=1, **kwargs):
        kwargs.pop("epochs", None)
        tf.keras.backend.clear_session()  # Clear previous models to free memory
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
        self.save_model_weights(epoch=self.num_epochs)
        return history

    ### === Saving and Loading Model Weights === ###
    
    def save_model_weights(self, epoch=1):
        """
        Saves the model weights inside 'weights_extraction', following the same structure as BERT.
        Example filename: weights/weights_extraction/roberta_epoch_1.h5
        """
        # Ensure the base directory exists
        os.makedirs(self.model_save_dir, exist_ok=True)
    
        # Define the save directory for weights
        save_dir = os.path.join(self.model_save_dir, "weights_extraction")
        os.makedirs(save_dir, exist_ok=True)
    
        # Define the paths for saving weights and zip file
        weights_path = os.path.join(save_dir, f"roberta_epoch_{epoch}.h5")
        zip_path = os.path.join(save_dir, f"roberta_epoch_{epoch}.zip")
    
        # Save the model weights
        self.model.save_weights(weights_path)
        print(f"Model weights saved to: {weights_path}")
    
        # Zip the weights file
        shutil.make_archive(weights_path.replace('.h5', ''), 'zip', save_dir)
        print(f"Model weights zipped and saved to: {zip_path}")
    
    def load_model_weights(self, epoch=1):
        """
        Loads the model weights from 'weights_extraction/roberta_epoch_{epoch}.h5'.
        """
        # Define the path to the weights file
        weights_path = os.path.join(self.model_save_dir, "weights_extraction", f"roberta_epoch_{epoch}.h5")
    
        # Check if the weights file exists
        if not os.path.exists(weights_path):
            raise ValueError(f"Model weights not found at {weights_path}")
    
        # Load the model weights
        self.model.load_weights(weights_path)
        print(f"Model weights loaded from: {weights_path}")

    ### === Extracting and Saving Feature Vectors === ###
    
    def get_features(self, X):
        """
        Extracts feature vectors from the RoBERTa model.
        Returns the [CLS] embedding from the last hidden layer.
        """
        return self.feature_extractor.predict(X)

    def extract_feature_vectors(self, X, split_name="train", data_num="1"):
        """
        Extracts and saves feature vectors.
        Saves to feature_vectors/{split_name}/roberta/roberta_{split_name}_{data_num}_features.npy and .txt
        """
        # Extract feature vectors
        features = self.get_features(X)
    
        # Define output directory (matching BERT's format)
        out_dir = os.path.join("feature_vectors", split_name, "roberta")
        os.makedirs(out_dir, exist_ok=True)
    
        # Save feature vectors in .npy format
        npy_path = os.path.join(out_dir, f"roberta_{split_name}_{data_num}_features.npy")
        np.save(npy_path, features)
        print(f"[{split_name.upper()}] Feature vectors saved to: {npy_path}")
    
        # Save feature vectors in .txt format
        txt_path = os.path.join(out_dir, f"roberta_{split_name}_{data_num}_features.txt")
        np.savetxt(txt_path, features, fmt='%.6f', delimiter=' ')
        print(f"[{split_name.upper()}] Feature vectors saved to: {txt_path}")

    def __getattr__(self, name):
        """
        Allows calls to underlying model methods except for 'predict', 'predict_proba', 'get_features', and 'extract_feature_vectors'.
        """
        if name not in ['predict', 'predict_proba', 'get_features', 'extract_feature_vectors']:
            return getattr(self.model, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
