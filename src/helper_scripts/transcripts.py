import csv
import json
import re
import pandas as pd

def parse_segments(transcript_id, transcript_text):
    """
    Split transcript_text into segments.
    Assumes each segment is on a new line and begins with a prefix ending with ':'.
    If the prefix is 'Interviewer', it is marked as an interviewer segment; otherwise, it is considered a candidate segment.
    For candidate segments, the speaker is always set to "Candidate".
    Both labels and sentiment fields are left empty for later update.
    """
    segments = []
    # Counters for interviewer and candidate segments
    interviewer_count = 1
    candidate_count = 1

    # Split transcript by newline; adjust if your CSV uses a different delimiter.
    lines = transcript_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Use regex to extract the prefix (speaker name) and the remaining text.
        match = re.match(r'^([^:]+):\s*(.*)', line)
        if match:
            speaker_raw, text = match.groups()
            # Check if the speaker is "Interviewer". Otherwise, treat it as a candidate segment.
            if speaker_raw.strip().lower() == "interviewer":
                speaker = "Interviewer"
                seg_id = f"{transcript_id}_i{interviewer_count:02d}"
                interviewer_count += 1
            else:
                speaker = "Candidate"  # regardless of the actual candidate name.
                seg_id = f"{transcript_id}_a{candidate_count:02d}"
                candidate_count += 1

            segment = {
                "segment_id": seg_id,
                "speaker": speaker,
                "text": text,
                "labels": {},       # empty labels
                "sentiment": ""     # empty sentiment
            }
            segments.append(segment)
        else:
            # If no prefix is found, append the line to the previous segment's text (if available).
            if segments:
                segments[-1]["text"] += " " + line
            else:
                # If it's the first line and no prefix, treat it as a candidate segment.
                seg_id = f"{transcript_id}_a{candidate_count:02d}"
                candidate_count += 1
                segments.append({
                    "segment_id": seg_id,
                    "speaker": "Candidate",
                    "text": line,
                    "labels": {},
                    "sentiment": ""
                })
    return segments

def csv_to_json(input_csv, output_json):
    # Read CSV file into a DataFrame.
    df = pd.read_csv(input_csv)

    results = []

    for index, row in df.iterrows():
        transcript_id = row["transcript_id"]
        jd_id = row["jd_id"]
        candidate_id = row["candidate_id"]
        transcript_text = row["transcript_text"]

        overall_scores = {
            "technical_skills": row["technical_skills"],
            "experience": row["experience"],
            "problem_solving": row["problem_solving"],
            "communication": row["communication"],
            "cultural_fit": row["cultural_fit"],
            "overall_suitability": row["overall_suitability"],
            "explanation": row["explaination"]  # using the CSV column name provided
        }

        segments = parse_segments(transcript_id, transcript_text)

        record = {
            "transcript_id": transcript_id,
            "jd_id": jd_id,
            "candidate_id": candidate_id,
            "overall_scores": overall_scores,
            "segments": segments
        }
        results.append(record)

    # Write the JSON output.
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    input_csv = "../../data/raw/transcripts/transcripts.csv"     # Replace with the path to your CSV file.
    output_json = "../../data/processed/transcriptions.json"  # Desired output JSON file.
    csv_to_json(input_csv, output_json)
    print(f"JSON file '{output_json}' has been created.")
