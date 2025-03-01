import pandas as pd

# Load the LIAR dataset (assumes it is in TSV format)
df = pd.read_csv("train_file.tsv", sep="\t", header=None)

# Assign column names based on LIAR dataset
df.columns = ["statement", "label", "subject", "speaker", "speaker_affiliation", 
              "context", "true_count", "mostly_true_count", "half_true_count",
              "mostly_false_count", "false_count", "pants_on_fire_count"]

# Select relevant metadata columns
metadata_cols = ["speaker", "speaker_affiliation", "context", 
                 "true_count", "mostly_true_count", "half_true_count", 
                 "mostly_false_count", "false_count", "pants_on_fire_count"]

# Extract metadata and save as CSV
metadata_df = df[metadata_cols]
metadata_df.to_csv("liar_metadata.csv", index=False)

print("✅ liar_metadata.csv created successfully!")

