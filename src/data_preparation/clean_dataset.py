import pandas as pd
import numpy as np

# Load the master dataset (assuming it has been saved as master_dataset.csv)
master_df = pd.read_csv('../../data/processed/master_dataset.csv')

# === 1. Data Consistency Checks ===

# Check that ID columns are strings
id_columns = ['transcript_id', 'jd_id', 'candidate_id', 'Questionnaire_ID', 'Transcript_ID']
for col in id_columns:
    if col in master_df.columns:
        master_df[col] = master_df[col].astype(str)

# Verify that Transcript_ID in questionnaires match transcript_id in master dataset
if 'Transcript_ID' in master_df.columns and 'transcript_id' in master_df.columns:
    inconsistent_transcript_ids = master_df[master_df['Transcript_ID'] != master_df['transcript_id']]
    print(f"Found {len(inconsistent_transcript_ids)} inconsistent Transcript_ID vs transcript_id records.")
    # Optionally, review or log these records:
    print(inconsistent_transcript_ids[['Transcript_ID', 'transcript_id']].head())

# Ensure that JD IDs match between JD data and transcript data:
if 'jd_id' in master_df.columns:
    jd_id_unique = master_df['jd_id'].unique()
    print(f"Number of unique JD IDs in master dataset: {len(jd_id_unique)}")
    
# === 2. Data Completeness Checks ===

# Define a list of critical fields that must not be missing
critical_fields = ['transcript_id', 'jd_id', 'candidate_id', 'segments', 
                   'overall_scores.technical_skills', 'overall_scores.experience',
                   'overall_scores.problem_solving', 'overall_scores.communication',
                   'overall_scores.cultural_fit', 'overall_scores.overall_suitability',
                   'jd_text', 'extracted_skills',
                   'Questionnaire_ID', 'Overall_Suitability', 'Recommendation', 'Interviewer_Comments']

# Check for missing values in critical fields
missing_summary = master_df[critical_fields].isnull().sum()
print("\nMissing values in critical fields:")
print(missing_summary)

# Handle missing values:
# Option 1: Remove rows with missing critical fields
initial_count = len(master_df)
master_df_clean = master_df.dropna(subset=critical_fields)
print(f"\nRemoved {initial_count - len(master_df_clean)} records due to missing critical fields.")

# Option 2 (Alternative): Impute missing values if appropriate (for numeric fields, for example)
# For example:
# master_df_clean['overall_scores.technical_skills'].fillna(master_df_clean['overall_scores.technical_skills'].mean(), inplace=True)

# === 3. Data Validation ===

# Manually inspect a random sample of cleaned records
sample_records = master_df_clean.sample(n=5, random_state=42)
print("\nSample records for manual inspection:")
print(sample_records[['transcript_id', 'jd_id', 'candidate_id', 'segments', 
                      'overall_scores.explanation', 'jd_text', 'extracted_skills',
                      'Questionnaire_ID', 'Overall_Suitability', 'Recommendation', 
                      'Interviewer_Comments']])

# Optionally, export the sample records to a CSV for detailed manual review
sample_records.to_csv('sample_validation_records.csv', index=False)
print("\nSample records saved to 'sample_validation_records.csv' for manual review.")

# === Optional Additional Checks ===

# Check if the number of segments per transcript seems reasonable:
def count_segments(segments):
    # Assuming segments are stored as a stringified JSON array; if already parsed, use len directly.
    try:
        import json
        seg_list = json.loads(segments)
        return len(seg_list)
    except Exception:
        # If segments is already a list, return its length; otherwise, return NaN
        if isinstance(segments, list):
            return len(segments)
        else:
            return np.nan

if 'segments' in master_df_clean.columns:
    master_df_clean['segment_count'] = master_df_clean['segments'].apply(count_segments)
    print("\nSegment count summary:")
    print(master_df_clean['segment_count'].describe())

# Save the cleaned master dataset for further processing
master_df_clean.to_csv('../../data/final/master_dataset_clean.csv', index=False)
print("\nCleaned master dataset saved as 'master_dataset_clean.csv'")
