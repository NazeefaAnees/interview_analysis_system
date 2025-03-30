import pandas as pd
from openai import OpenAI
import json
import time
import re

# Set your OpenAI API key
openai_api_key = ""
openai_client = OpenAI(api_key=openai_api_key)

# Configuration: batch size and maximum number of retries for each API call.
BATCH_SIZE = 5
MAX_RETRIES = 3

def clean_json_output(text):
    """
    Extract the valid JSON portion from the API response.
    Uses a regex to find the first valid JSON object or array.
    """
    json_match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    return text

def annotate_batch(batch_records):
    """
    Annotate a batch of records. Each record contains JD details and transcript.
    Returns a list of annotation dictionaries in the same order.
    """
    prompt = f"""
You are a highly experienced recruitment assessment assistant. Your task is to evaluate interview transcripts for a Software Engineering position by combining insights from both the job description and the candidate's transcript. Treat each evaluation as if you were the interviewer filling out an assessment form.

For each evaluation, use the following definitions and rating scales (1=poor, 5=excellent):

- Technical Proficiency: Evaluate the candidate's coding skills, familiarity with tools, and system design.
- Problem Solving Ability: Assess the candidate's approach to identifying and solving complex technical problems.
- Communication Skills: Evaluate clarity, organization, and effectiveness in conveying ideas.
- Cultural Team Fit: Consider alignment with company values, teamwork, and adaptability.
- Adaptability & Learning: Assess willingness and ability to learn new skills and adapt to challenges.

Also, provide an overall suitability rating (1-5), a recommendation (choose from "Hire", "Consider", or "Reject"), and write a short, human-like summary in the interviewer comments explaining key strengths and areas for improvement.

For each evaluation, generate a unique Questionnaire_ID in sequential order, formatted as "q001", "q002", etc.

Strictly follow the format below for each evaluation. Do not include any other text or formatting outside the JSON array.

Return the results for all evaluations as a single JSON array of objects in exactly the following format with no additional text:

{{
  "Questionnaire_ID": "<Questionnaire_ID>",
  "Transcript_ID": "<Transcript_ID>",
  "Role": "<Role>",
  "Ratings": {{
    "Technical_Proficiency": <number>,
    "Problem_Solving_Ability": <number>,
    "Communication_Skills": <number>,
    "Cultural_Team_Fit": <number>,
    "Adaptability_Learning": <number>
  }},
  "Overall_Suitability": <number>,
  "Recommendation": "<Hire/Consider/Reject>",
  "Interviewer_Comments": "<comments>"
}}

Below are the evaluations to complete:
"""
    # Append details for each record.
    for rec in batch_records:
        transcript_id = rec["transcript_id"]
        role = rec["role"]
        jd_text = rec["jd_text"]
        transcript_text = rec["transcript_text"]
        prompt += f'\nTranscript_ID: "{transcript_id}"\n'
        prompt += f'Role: "{role}"\n'
        prompt += f"Job Description: \"{jd_text}\"\n"
        prompt += f"Transcript: \"{transcript_text}\"\n"
    
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = openai_client.chat.completions.create(
                    model="o1-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                            ],
                        }
                    ], 
                     
                )
            answer_text = response.choices[0].message.content
            cleaned_text = clean_json_output(answer_text)
            annotations = json.loads(cleaned_text)
            print(f"Received annotations for {len(annotations)} records.")
            print(f"Annotations: {annotations}")
            if isinstance(annotations, list):
                # Ensure each object has a Transcript_ID field.
                if all("Transcript_ID" in ann for ann in annotations):
                    return annotations
            raise ValueError("Unexpected output format.")
        except Exception as e:
            retries += 1
            print(f"Batch retry {retries}/{MAX_RETRIES} due to error: {e}")
            time.sleep(2)
    return []

def main(jd_csv, transcript_csv, output_json):
    # Read the JD CSV file.
    jd_df = pd.read_csv(jd_csv)
    # Read the transcript CSV file.
    transcript_df = pd.read_csv(transcript_csv)

    # Merge transcripts with corresponding JD info on jd_id.
    merged_df = transcript_df.merge(jd_df, on="jd_id", how="left", suffixes=("", "_jd"))
    # Use only the necessary columns.
    records = merged_df[["transcript_id", "role", "jd_text", "transcript_text"]].to_dict(orient="records")
    
    all_annotations = []
    # Process records in batches.
    for i in range(0, len(records), BATCH_SIZE):
        batch_records = records[i:i+BATCH_SIZE]
        print(f"Processing batch {i // BATCH_SIZE + 1} with {len(batch_records)} records...")
        batch_annotations = annotate_batch(batch_records)
        all_annotations.extend(batch_annotations)
        time.sleep(1)

    # Reassign Questionnaire_IDs sequentially (q001, q002, ...) while preserving Transcript_ID.
    for idx, ann in enumerate(all_annotations, start=1):
        ann["Questionnaire_ID"] = f"q{idx:03d}"
    
    # Write the aggregated annotations to the output JSON file.
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, indent=2)
    print(f"Annotated questionnaires saved to {output_json}")

if __name__ == "__main__":
    jd_csv = "../../data/processed/job_descriptions.csv"                   # CSV with columns: jd_id, role, jd_text, extracted_skills
    transcript_csv = "../../data/raw/transcripts/transcripts.csv"  # CSV with columns: transcript_id, jd_id, candidate_id, transcript_text, technical_skills, experience, problem_solving, communication, cultural_fit, overall_suitability, explaination
    output_json = "../../data/processed/questionnaires.json"
    main(jd_csv, transcript_csv, output_json)
