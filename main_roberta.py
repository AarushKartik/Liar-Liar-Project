from src.download_data import download

# Roberta
from src.preprocessing_roberta import process_data_pipeline_roberta
from src.roberta_model import Roberta
from src.train_roberta import train_roberta

def roberta():
    print("Starting the pipeline...")

    # Stage 1: Download the Liar Dataset
    print("Step 1: Downloading the dataset...")
    download()
    print("Dataset downloaded successfully.\n")

    # Stage 2: Process the data
    print("Step 2: Preprocessing the data...")
    X_train, y_train, X_test, y_test = process_data_pipeline('train.tsv', 'test.tsv', 'valid.tsv')
    print("Data preprocessing complete.\n")

    # Stage 3: Gather the model
    print("Step 3: Building the RNN model...")
    from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
    def build_roberta_model():
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
        return model


    # Stage 4: Train the model
    def train_roberta_model(train_dataset, test_dataset, model):
    training_args = TrainingArguments(
        output_dir='./results',          # Output directory for model checkpoints
        num_train_epochs=3,              # Number of training epochs
        per_device_train_batch_size=8,   # Batch size per device during training
        per_device_eval_batch_size=16,   # Batch size per device during evaluation
        warmup_steps=500,                # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # Strength of weight decay
        logging_dir='./logs',            # Directory for storing logs
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,                         # The RoBERTa model
        args=training_args,                  # Training arguments
        train_dataset=train_dataset,         # Training dataset
        eval_dataset=test_dataset,           # Evaluation dataset
        tokenizer=tokenizer,                 # Tokenizer for encoding
    )

    # Train the model
    trainer.train()



if __name__ == '__main__':
    
    roberta()
