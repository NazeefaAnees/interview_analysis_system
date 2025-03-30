from openai import OpenAI
import json
import time
import re

# Set your OpenAI API key
openai_api_key = ""
openai_client = OpenAI(api_key=openai_api_key)
# Define batch size (number of candidate segments per API call)
BATCH_SIZE = 5
MAX_RETRIES = 3



def clean_json_output(text):
    """
    Attempt to extract a valid JSON string from the API response text.
    It removes any leading/trailing text outside the outermost JSON structure.
    """
    json_match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    return text

def annotate_segments_batch(segments_batch):
    """
    Annotate a batch of candidate segments.
    segments_batch is a list of dictionaries, each with keys "segment_id" and "text".
    Returns a list of annotation dictionaries in the same order.
    """
    prompt = f"""
You are an annotation assistant. For each provided interview segment from a candidate, annotate with the following keys:
- "segment_id": the segment identifier (provided).
- "technical_skills": (true/false) True if the segment mentions specific technologies, programming languages, or system design.
- "problem_solving": (true/false) True if the segment demonstrates structured or innovative problem-solving.
- "communication": (true/false) True if the segment shows clear, organized, and persuasive language.
- "experience": (true/false) True if the segment mentions work experience, years of work, or leadership.
- "cultural_fit": (true/false) True if the segment indicates teamwork, adaptability, or alignment with company values.
- "sentiment": a string that is one of "positive", "neutral", or "negative" based on the tone of the segment.

For example, if a segment is: 
Segment ID: "t001_a01"
Text: "I've been working in fintech for 6 years, leading teams on ML projects using Python, SQL, and LightGBM."
Then a valid annotation would be:
{{
  "segment_id": "t001_a01",
  "technical_skills": true,
  "problem_solving": false,
  "communication": true,
  "experience": true,
  "cultural_fit": false,
  "sentiment": "positive"
}}

IMPORTANT:
- Process all candidate segments provided and return a single JSON array containing one JSON object per segment.
- STRICTLY Do not output any extra text, explanations, or markdown formatting. Only return the JSON array.
- STRICTLY Ensure the output is valid JSON.
- STRICTLY Do not include any other text or formatting outside the JSON array.

Here are the candidate segments to annotate:
"""

    # Append each candidate segment
    for seg in segments_batch:
        prompt += f'\nSegment ID: "{seg["segment_id"]}"\nText: "{seg["text"]}"\n'
    
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
            print(f"Received annotations for {len(annotations)} segments.")
            print(f"Annotations: {annotations}")
            if isinstance(annotations, list) and all("segment_id" in ann for ann in annotations):
                return annotations
            else:
                raise ValueError("Response JSON does not match expected format.")
        except Exception as e:
            retries += 1
            print(f"Retry {retries}/{MAX_RETRIES} due to error: {e}")
            time.sleep(2)
    return []

def annotate_transcripts(input_json, output_json):
    """
    Reads the input JSON file with transcripts and segments, batches only candidate segments
    for annotation, and writes the updated JSON file with annotations.
    """
    with open(input_json, "r", encoding="utf-8") as f:
        transcripts = json.load(f)

    for transcript in transcripts:
        segments = transcript.get("segments", [])
        # Separate candidate segments (for annotation) from interviewer segments.
        candidate_segments = [seg for seg in segments if seg.get("speaker") == "Candidate"]
        non_candidate_segments = [seg for seg in segments if seg.get("speaker") != "Candidate"]

        # Annotate candidate segments in batches.
        annotated_candidate_segments = []
        for i in range(0, len(candidate_segments), BATCH_SIZE):
            batch = candidate_segments[i:i+BATCH_SIZE]
            annotations = annotate_segments_batch(batch)
            ann_dict = {ann["segment_id"]: ann for ann in annotations}
            for seg in batch:
                seg_ann = ann_dict.get(seg["segment_id"], {})
                seg["labels"] = {
                    "technical_skills": seg_ann.get("technical_skills", None),
                    "problem_solving": seg_ann.get("problem_solving", None),
                    "communication": seg_ann.get("communication", None),
                    "experience": seg_ann.get("experience", None),
                    "cultural_fit": seg_ann.get("cultural_fit", None)
                }
                seg["sentiment"] = seg_ann.get("sentiment", "")
                annotated_candidate_segments.append(seg)
            time.sleep(1)
        
        # Combine annotated candidate segments with non-candidate segments.
        transcript["segments"] = annotated_candidate_segments + non_candidate_segments

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, indent=2)

if __name__ == "__main__":
    input_json = "../../data/processed/transcriptions.json"             # Your structured transcript JSON file.
    output_json = "../../data/processed/annotated_transcriptions.json"   # The JSON file with annotations.
    annotate_transcripts(input_json, output_json)
    print(f"Annotated transcripts saved to {output_json}")
