import pandas as pd
import json

# Step 1: Load the JD CSV
jd_df = pd.read_csv('../../data/processed/job_descriptions.csv')
print("JD DataFrame:")
print(jd_df.head())

# Step 2: Load the transcript annotations JSON
with open('../../data/processed/annotated_transcriptions.json', 'r', encoding='utf-8') as f:
    transcripts_data = json.load(f)

# Convert the list of transcript records to a DataFrame
transcripts_df = pd.json_normalize(transcripts_data)
print("\nTranscripts DataFrame:")
print(transcripts_df.head())

# Step 3: Load the questionnaires JSON
with open('../../data/processed/questionnaires.json', 'r', encoding='utf-8') as f:
    questionnaires_data = json.load(f)

# Convert the list of questionnaire records to a DataFrame
questionnaires_df = pd.json_normalize(questionnaires_data)
print("\nQuestionnaires DataFrame:")
print(questionnaires_df.head())

# Step 4: Merge JD with Transcripts on 'jd_id'
# Ensure that 'jd_id' columns in both dataframes are of the same type
transcripts_df['jd_id'] = transcripts_df['jd_id'].astype(str)
jd_df['jd_id'] = jd_df['jd_id'].astype(str)

merged_df = pd.merge(transcripts_df, jd_df, on='jd_id', how='left')
print("\nMerged JD and Transcripts DataFrame:")
print(merged_df.head())

# Step 5: Merge the above with Questionnaires on Transcript_ID
# Make sure the transcript ID fields match; here we assume:
#   - transcripts_df uses "transcript_id"
#   - questionnaires_df uses "Transcript_ID"
merged_df['transcript_id'] = merged_df['transcript_id'].astype(str)
questionnaires_df['Transcript_ID'] = questionnaires_df['Transcript_ID'].astype(str)

# Merge on transcript ID (left join to keep all transcript records)
master_df = pd.merge(merged_df, questionnaires_df, left_on='transcript_id', right_on='Transcript_ID', how='left')
print("\nMaster DataFrame (Merged with Questionnaires):")
print(master_df.head())

# Optional: Save the master DataFrame to CSV for later use
master_df.to_csv('../../data/processed/master_dataset.csv', index=False)
print("\nMaster dataset saved to 'master_dataset.csv'")
