import pandas as pd
import numpy as np
import json
import ast
from sentence_transformers import SentenceTransformer

# Load the pre-trained SentenceTransformer model (lightweight and CPU-friendly)
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = model.get_sentence_embedding_dimension()

# Randomly initialize an attention parameter vector (for demonstration)
w = np.random.randn(embedding_dim)

def sentiment_to_score(sentiment):
    """
    Map a sentiment string to a numeric score.
    """
    mapping = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}
    # Convert sentiment to lowercase and lookup; default to 0.5 if not recognized
    if isinstance(sentiment, str):
        return mapping.get(sentiment.lower(), 0.5)
    return 0.5

def aggregate_embeddings_attention(segments_field, embedding_model, w):
    """
    Given a segments field (as a JSON string or Python list), compute:
      - The embedding for each segment using a pre-trained transformer.
      - An attention score for each segment via a dot product with parameter vector w.
      - Adjust attention scores with sentiment values.
      - Compute a softmax over these scores.
      - Return the attention-weighted aggregated embedding and the attention weights.
    """
    # Attempt to parse segments_field using ast.literal_eval if necessary
    segments = None
    if isinstance(segments_field, str):
        try:
            segments = json.loads(segments_field)
        except Exception:
            try:
                segments = ast.literal_eval(segments_field)
            except Exception as e:
                print(f"Error parsing segments: {e}")
                return None, None
    else:
        segments = segments_field

    # Extract texts and sentiments from segments
    texts = [seg.get("text", "") for seg in segments]
    sentiments = [seg.get("sentiment", "neutral") for seg in segments]

    # Compute embeddings for each segment (list of numpy arrays)
    embeddings = embedding_model.encode(texts, show_progress_bar=False)

    # Compute raw attention scores using dot product between each embedding and w
    scores = np.array([np.dot(embed, w) for embed in embeddings])
    
    # Convert sentiment strings to numeric sentiment scores
    sentiment_scores = np.array([sentiment_to_score(s) for s in sentiments])
    
    # Combine attention scores with sentiment (multiplicative combination)
    combined_scores = scores * sentiment_scores

    # Compute softmax weights for stability
    exp_scores = np.exp(combined_scores - np.max(combined_scores))
    attn_weights = exp_scores / np.sum(exp_scores) if np.sum(exp_scores) > 0 else np.ones_like(exp_scores)/len(exp_scores)
    
    # Compute the attention-weighted sum of the segment embeddings
    aggregated_embedding = np.sum(attn_weights[:, np.newaxis] * embeddings, axis=0)
    
    return aggregated_embedding, attn_weights

# --- Example Usage on a Master Dataset ---

# Load your cleaned master dataset (adjust the filename as needed)
df = pd.read_csv('../../data/final/final_dataset.csv')

# Apply the attention-based aggregation for each transcript.
# Here, we assume the 'segments' column contains the JSON string for segments.
aggregated_embeddings = []
attention_weights_list = []

for idx, row in df.iterrows():
    seg_field = row['segments']
    agg_embed, attn_weights = aggregate_embeddings_attention(seg_field, model, w)
    # If aggregation fails, substitute a zero vector
    if agg_embed is None:
        agg_embed = np.zeros(embedding_dim)
    aggregated_embeddings.append(agg_embed)
    attention_weights_list.append(attn_weights)

# Optionally, add the aggregated embeddings to the DataFrame (store as JSON strings)
df['transcript_embedding'] = [json.dumps(embed.tolist()) for embed in aggregated_embeddings]

# Save the updated master dataset with aggregated embeddings
output_file = '../../data/test-final/master_dataset_with_attention_embeddings.csv'
df.to_csv(output_file, index=False)
print(f"Aggregated transcript embeddings (with attention) saved to '{output_file}'")
