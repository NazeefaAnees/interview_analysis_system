import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
import shap
import joblib
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import re
import shap
from shap import Explanation
# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR,"src", "models", "saved")
MASTER_CSV = os.path.join(BASE_DIR, "data","test-final", "FINAL_master.csv")
JD_CSV = os.path.join(BASE_DIR, "data","test-final", "jds_clean.csv")

# --- Load models ---
reg_concat = joblib.load(os.path.join(MODEL_DIR, "full_regressor.joblib"))
clf_concat = joblib.load(os.path.join(MODEL_DIR, "full_classifier.joblib"))


# --- Load skill2idx ---
with open(os.path.join(MODEL_DIR, "skill2idx.json"), "r") as f:
    skill2idx = json.load(f)
skill_dim = len(skill2idx) + 1# Match training data's skills_vec (34 dimensions)



# --- Load HF model ---
HF_MODEL = "distilbert-base-uncased"
try:
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    hf_model = AutoModel.from_pretrained(HF_MODEL)
    print(f"Debug: Loaded {HF_MODEL} tokenizer and model")
    print(f"Debug: Loaded {HF_MODEL} tokenizer and model")
except Exception as e:
    st.error(f"Error loading Hugging Face model: {str(e)}")
    print(f"Error: Failed to load Hugging Face model: {str(e)}")
    st.stop()



class InterviewDataset(Dataset):
    def __init__(self, master_csv, jd_csv):
        df = pd.read_csv(master_csv)
        jd = pd.read_csv(jd_csv)
        df = df.merge(jd[['jd_id','jd_text_embedding']], on='jd_id', how='left')

        def parse_emb(s):
            try:
                return np.array(json.loads(s))
            except:
                parts = re.split(r'[,\s]+', s.strip().lstrip('[').rstrip(']'))
                return np.array([float(x) for x in parts if x])

        def parse_skills(s):
            return np.array(json.loads(s.replace("'", '"')))

        # 1) Define the interviewer-rating columns
        rating_cols = [
            'Ratings.Technical_Proficiency',
            'Ratings.Problem_Solving_Ability',
            'Ratings.Communication_Skills',
            'Ratings.Cultural_Team_Fit',
            'Ratings.Adaptability_Learning'
        ]
        # 2) Ensure they're numeric
        df[rating_cols] = df[rating_cols].apply(pd.to_numeric, errors='coerce').fillna(0.)

        # 3) Build your struct vector as: [skills_vec, segment_count, *ratings_vec]
        df['struct'] = df.apply(lambda r: np.concatenate([
            parse_skills(r['skills_vector']),
            [r['segment_count']],
            r[rating_cols].values.astype(float)
        ]), axis=1)

        # the rest stays the same…
        df['text'] = df['transcript_embedding'].apply(parse_emb)
        df['jd']   = df['jd_text_embedding'].  apply(parse_emb)

        df = df.dropna(subset=['struct','text','jd'])
        self.s = np.vstack(df['struct'].values).astype(np.float32)
        self.t = np.vstack(df['text'].  values).astype(np.float32)
        self.j = np.vstack(df['jd'].    values).astype(np.float32)

        self.y_reg = df['Overall_Score'].values.astype(np.float32)
        rec_cols   = ['rec_Hire','rec_Consider','rec_Reject']
        self.y_cls = df[rec_cols].values.argmax(axis=1).astype(np.int64)

    def __len__(self):
        return len(self.y_reg)

    def __getitem__(self, idx):
        return (self.s[idx], self.j[idx], self.t[idx],
                self.y_reg[idx], self.y_cls[idx])


# === 1. Gated Fusion Model ===
class GatedFusionModel(nn.Module):
    def __init__(self, dim_s, dim_j, dim_t, hidden, dropout):
        super().__init__()
        self.proj_s = nn.Sequential(nn.Linear(dim_s, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.proj_j = nn.Sequential(nn.Linear(dim_j, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.proj_t = nn.Sequential(nn.Linear(dim_t, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.gate_s = nn.Linear(dim_s, hidden)
        self.gate_j = nn.Linear(dim_j, hidden)
        self.gate_t = nn.Linear(dim_t, hidden)
        self.head_reg = nn.Linear(hidden, 1)
        self.head_cls = nn.Linear(hidden, 3)

    def forward(self, s, j, t):
        hs = self.proj_s(s)
        hj = self.proj_j(j)
        ht = self.proj_t(t)
        gs = torch.sigmoid(self.gate_s(s))
        gj = torch.sigmoid(self.gate_j(j))
        gt = torch.sigmoid(self.gate_t(t))
        fused = gs * hs + gj * hj + gt * ht
        return fused

    def predict(self, s, j, t):
        fused = self.forward(s, j, t)
        return self.head_reg(fused).squeeze(-1), self.head_cls(fused)
    
# --- Cross‑Attention Fusion Model ---
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim_s, dim_j, dim_t, hidden, heads, dropout):
        super().__init__()
        # Projections
        self.proj_s = nn.Sequential(nn.Linear(dim_s, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.proj_j = nn.Sequential(nn.Linear(dim_j, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.proj_t = nn.Sequential(nn.Linear(dim_t, hidden), nn.ReLU(), nn.Dropout(dropout))
        # Self‑attention over the 3 modality tokens
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=heads, dropout=dropout, batch_first=True)
        # Prediction heads
        self.head_reg = nn.Linear(hidden, 1)
        self.head_cls = nn.Linear(hidden, 3)

    def forward(self, s, j, t):
        hs = self.proj_s(s).unsqueeze(1)  # [B,1,H]
        hj = self.proj_j(j).unsqueeze(1)
        ht = self.proj_t(t).unsqueeze(1)
        seq = torch.cat([hs, hj, ht], dim=1)  # [B,3,H]
        attn_out, _ = self.attn(seq, seq, seq)
        fused = attn_out.mean(dim=1)  # [B,H]
        return fused

    def predict(self, s, j, t):
        fused = self.forward(s, j, t)
        return self.head_reg(fused).squeeze(-1), self.head_cls(fused)
    
dataset = InterviewDataset(MASTER_CSV, JD_CSV)
dim_s, dim_j, dim_t = dataset.s.shape[1], dataset.j.shape[1], dataset.t.shape[1]

print("Debug: Initialized placeholder gated and attn models")
gated = GatedFusionModel(dim_s, dim_j, dim_t, hidden=256, dropout=0.1)
gated.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "gated_model.pth"), map_location="cpu")
)
gated.eval()

attn = CrossAttentionFusion(dim_s, dim_j, dim_t, hidden=256, heads=2, dropout=0.45)
attn.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "cross_attention_model.pth"), map_location="cpu")
)
attn.eval()




# --- Parse skills ---
def parse_skills(s: str) -> np.ndarray:
    print(f"Debug: Parsing skills input: {s}")
    print(f"Debug: Parsing skills input: {s}")
    vec = np.zeros(skill_dim, dtype=np.float32)
    skills = [w.strip().lower() for w in s.split(",")]
    matched_skills = []
    for kw in skills:
        if kw in skill2idx:
            vec[skill2idx[kw]] = 1.0
            matched_skills.append(kw)
    print(f"Debug: Matched skills: {matched_skills}")
    print(f"Debug: Skills vector shape: {vec.shape}, non-zero indices: {np.nonzero(vec)[0]}")
    print(f"Debug: Matched skills: {matched_skills}")
    print(f"Debug: Skills vector shape: {vec.shape}, non-zero indices: {np.nonzero(vec)[0]}")
    return vec

# --- Get embedding (reduce to 384 dimensions, return 2D) ---
def get_embedding(text: str, target_dim: int) -> np.ndarray:
    toks = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        out = hf_model(**toks)
    emb = out.last_hidden_state.mean(dim=1).numpy()  # (1, hidden_size)
    # now pad or truncate to exactly target_dim
    if emb.shape[1] > target_dim:
        emb = emb[:, :target_dim]
    elif emb.shape[1] < target_dim:
        emb = np.pad(emb, ((0,0),(0,target_dim-emb.shape[1])), mode="constant")
    return emb  # still shape (1, target_dim)

# --- Run inference ---
def run_inference(s_vec: np.ndarray, transcript_txt: str, jd_text: str, review_threshold: float = 0.6):
    print("Debug: Starting inference")
    print("Debug: Starting inference")
    t_emb = get_embedding(transcript_txt, dim_t)
    j_emb = get_embedding(jd_text, dim_j)

    feat = np.hstack([s_vec, j_emb, t_emb])
    print(f"Debug: Feature shapes - s_vec: {s_vec.shape}, j_emb: {j_emb.shape}, t_emb: {t_emb.shape}")
    print(f"Debug: Concatenated feats shape: {feat.shape}")
    print(f"Debug: Feature shapes - s_vec: {s_vec.shape}, j_emb: {j_emb.shape}, t_emb: {t_emb.shape}")
    print(f"Debug: Concatenated feats shape: {feat.shape}")

    if feat.shape[1] != reg_concat.n_features_in_:
        raise ValueError(f"Feature dimension mismatch: got {feat.shape[1]}, expected {reg_concat.n_features_in_}")
    print("Debug: Feature dimension matches expected 808")
    print("Debug: Feature dimension matches expected 808")

    r0 = reg_concat.predict(feat)[0]
    p0 = clf_concat.predict_proba(feat)[0]
    print(f"Debug: Concat model - regression: {r0:.2f}, classification probs: {p0}")
    print(f"Debug: Concat model - regression: {r0:.2f}, classification probs: {p0}")

    s_t = torch.tensor(s_vec, dtype=torch.float32)
    j_t = torch.tensor(j_emb, dtype=torch.float32)
    t_t = torch.tensor(t_emb, dtype=torch.float32)
    r1, p1_logits = gated.predict(s_t, j_t, t_t)
    p1 = F.softmax(p1_logits, dim=1).detach().numpy()[0]
    r1 = r1.detach().numpy()[0]
    print(f"Debug: Gated model - regression: {r1:.2f}, classification probs: {p1}")
    print(f"Debug: Gated model - regression: {r1:.2f}, classification probs: {p1}")

    r2, p2_logits = attn.predict(s_t, j_t, t_t)
    p2 = F.softmax(p2_logits, dim=1).detach().numpy()[0]
    r2 = r2.detach().numpy()[0]
    print(f"Debug: Attn model - regression: {r2:.2f}, classification probs: {p2}")
    print(f"Debug: Attn model - regression: {r2:.2f}, classification probs: {p2}")

    r_ens = float((r0 + r1 + r2) / 3.0)
    p_ens = (p0 + p1 + p2) / 3.0
    print(f"Debug: Ensemble - regression: {r_ens:.2f}, classification probs: {p_ens}")
    print(f"Debug: Ensemble - regression: {r_ens:.2f}, classification probs: {p_ens}")

    max_prob = p_ens.max()
    needs_review = bool(max_prob < review_threshold)

    return {
        "overall_score": r_ens,
        "hire_prob": p_ens[0],
        "consider_prob": p_ens[1],
        "reject_prob": p_ens[2],
        "needs_review": needs_review,
        "review_reason": ("low confidence" if needs_review else ""),
        "raw_probs": p_ens,
    }

# --- Placeholder for log_for_review ---
def log_for_review(data):
    print("Debug: Logging for review:", data)
    print("Debug: Logging for review:", data)

# --- Color recommendation function ---
def color_recommendation(val):
    color = 'green' if val == 'Hire' else 'yellow' if val == 'Consider' else 'red'
    return f'background-color: {color}'

# --- Streamlit UI ---
st.title("🎯 Candidate Assessment Dashboard")
st.markdown("Evaluate candidates based on skills, job descriptions, and transcripts.")


# Sidebar
with st.sidebar:
    st.header("📋 Input Parameters")
    skills_input = st.text_input("Skills (comma-separated)", "Machine Learning, Python, SQL")
    transcript_txt = st.text_area("Transcript", "Sample transcript text...")
    jd_text = st.text_area("Job Description", "Data Scientist role...")
    segment_count = st.number_input("Segment Count", min_value=1, value=20)
    tech = st.slider("Technical Proficiency", 0.0, 1.0, 1.0)
    prob = st.slider("Problem Solving", 0.0, 1.0, 1.0)
    comm = st.slider("Communication Skills", 0.0, 1.0, 1.0)
    cult = st.slider("Cultural Fit", 0.0, 1.0, 1.0)
    adapt = st.slider("Adaptability", 0.0, 1.0, 1.0)

    
    rating_cols = [
        'Technical_Proficiency',
        'Problem_Solving_Ability',
        'Communication_Skills',
        'Cultural_Team_Fit',
        'Adaptability_Learning'
    ]
    dim_j, dim_t = dataset.j.shape[1], dataset.t.shape[1]

    feature_names = (
        list(skill2idx.keys())+ ["<placeholder>"]            # your skill flags, length=skill_dim
    + ["segment_count"]                 # +1
    + rating_cols                       # +5
    + [f"jd_emb_{i}"   for i in range(dim_j)]
    + [f"text_emb_{i}" for i in range(dim_t)]
    )


# Run Assessment
if st.sidebar.button("Run Assessment"):
    try:
        print("Debug: Building s_vec")
        print("Debug: Building s_vec")
        s_vec = np.concatenate([
            parse_skills(skills_input),
            [segment_count],
            [tech,prob,comm,cult,adapt]
        ]).reshape(1, -1)       
        print(f"Debug: s_vec shape: {s_vec.shape}")
        print(f"Debug: s_vec shape: {s_vec.shape}")

        print("Debug: Generating j_vec")
        print("Debug: Generating j_vec")
        j_vec = get_embedding(jd_text,dim_j)
        print(f"Debug: j_vec shape: {j_vec.shape}")
        print(f"Debug: j_vec shape: {j_vec.shape}")

        print("Debug: Generating t_vec")
        print("Debug: Generating t_vec")
        t_vec = get_embedding(transcript_txt,dim_t)
        print(f"Debug: t_vec shape: {t_vec.shape}")
        print(f"Debug: t_vec shape: {t_vec.shape}")

        print("Debug: Concatenating features")
        print("Debug: Concatenating features")
        feats = np.hstack([s_vec, j_vec, t_vec])
        
        print(f"Debug: feats shape: {feats.shape}")
        print(f"Debug: feats shape: {feats.shape}")

        if feats.shape[1] != reg_concat.n_features_in_:
            st.error(f"Feature dimension mismatch: got {feats.shape[1]}, expected {reg_concat.n_features_in_}")
            print(f"Debug: s_vec shape: {s_vec.shape}")
            print(f"Debug: j_vec shape: {j_vec.shape}")
            print(f"Debug: t_vec shape: {t_vec.shape}")
            print(f"Error: Feature dimension mismatch: got {feats.shape[1]}, expected {reg_concat.n_features_in_}")
        else:
            print("Debug: Feature dimension matches expected 808")
            print("Debug: Feature dimension matches expected 808")
            result = run_inference(s_vec, transcript_txt, jd_text)

            st.header("📊 Assessment Results")
            st.subheader("🔢 Scores")
            score_df = pd.DataFrame({
                "Model": ["Concat", "Gated", "Cross-Attn", "Ensemble"],
                "Score": [result['overall_score'], 0.0, 0.0, result['overall_score']]
            })
            st.dataframe(score_df.style.format({"Score": "{:.2f}"}))

            st.subheader("🎯 Recommendation")
            rec = ['Hire', 'Consider', 'Reject'][np.argmax(result['raw_probs'])]
            prob_df = pd.DataFrame({
                "Recommendation": ["Hire", "Consider", "Reject"],
                "Probability": [result['hire_prob'], result['consider_prob'], result['reject_prob']]
            })
            styled_df = prob_df.style.format({"Probability": "{:.2f}"}).map(color_recommendation, subset=["Recommendation"])
            st.dataframe(styled_df)

            if result['needs_review']:
                st.warning("⚠️ **Low confidence** ─ Recommended for review")
                log_for_review({
                    "jd": jd_text,
                    "transcript": transcript_txt,
                    "skills": skills_input,
                    "segment_count": segment_count,
                    "ratings": [tech, prob, comm, cult, adapt],
                    "ensemble_conf": float(result['raw_probs'].max()),
                    "ensemble_rec": rec
                })

            feats_df = pd.DataFrame(feats, columns=feature_names)

            rating_cols = [
                "Technical_Proficiency",
                "Problem_Solving_Ability",
                "Communication_Skills",
                "Cultural_Team_Fit",
                "Adaptability_Learning",
            ]
            struct_cols = list(skill2idx.keys()) + ["segment_count"] + rating_cols
            struct_df   = feats_df[struct_cols]

            # 2) initialize TreeExplainer and disable the additivity check
            explainer = shap.TreeExplainer(reg_concat)

            # 3) compute shap values
            shap_exp = explainer(
                struct_df,
                check_additivity=False
            )

            # 4) render only the top 5 bars inside its own expander
            with st.expander("🧠 SHAP — Top 5 Structured Features"):
                shap.plots.bar(shap_exp, show=False, max_display=5)
                plt.tight_layout()
                st.pyplot(plt)
                plt.clf()
                        
                        

            with st.expander("🔀 Gate Activations"):
                # turn them into Torch tensors again
                s_t = torch.tensor(s_vec, dtype=torch.float32)
                j_t = torch.tensor(j_vec, dtype=torch.float32)
                t_t = torch.tensor(t_vec, dtype=torch.float32)

                # apply the gating layers directly:
                g_s = torch.sigmoid(gated.gate_s(s_t)).mean().item()
                g_j = torch.sigmoid(gated.gate_j(j_t)).mean().item()
                g_t = torch.sigmoid(gated.gate_t(t_t)).mean().item()

                st.write({
                    "Struct → gate": round(g_s, 3),
                    "JD     → gate": round(g_j, 3),
                    "Text   → gate": round(g_t, 3),
                })


    except Exception as e:
        st.error(f"Error during assessment: {str(e)}")
        print(f"Error: Assessment failed: {str(e)}")

# Test with Sample Data
if st.sidebar.button("Test with Sample Data"):
    try:
        print("Debug: Loading 3 random rows from FINAL_master.csv")
        print("Debug: Loading 3 random rows from FINAL_master.csv")
        sample = pd.read_csv(MASTER_CSV).sample(n=1, random_state=42)
        jd_df = pd.read_csv(JD_CSV)
        print(f"Debug: Sampled {len(sample)} rows")
        print(f"Debug: Sampled {len(sample)} rows")

        # Summary table
        summary_data = []
        progress_bar = st.progress(0)
        total_rows = len(sample)

        for i, (idx, row) in enumerate(sample.iterrows()):
            print(f"Debug: Processing row {idx}")
            print(f"Debug: Processing row {idx}")
            skills_input = row['extracted_skills']
            jd_text = jd_df[jd_df['jd_id'] == row['jd_id']]['jd_text_clean'].iloc[0]
            transcript_txt = row.get('segments', 'Sample transcript text')
            segment_count = row['segment_count']
            ratings = row[['Ratings.Technical_Proficiency', 'Ratings.Problem_Solving_Ability',
                           'Ratings.Communication_Skills', 'Ratings.Cultural_Team_Fit',
                           'Ratings.Adaptability_Learning']].values
            tech, prob, comm, cult, adapt = ratings
            print(f"Debug: Row {idx} - skills: {skills_input}, jd_id: {row['jd_id']}, segment_count: {segment_count}")
            print(f"Debug: Row {idx} - Ratings - tech: {tech}, prob: {prob}, comm: {comm}, cult: {cult}, adapt: {adapt}")
            print(f"Debug: Row {idx} - skills: {skills_input}, jd_id: {row['jd_id']}, segment_count: {segment_count}")
            print(f"Debug: Row {idx} - Ratings - tech: {tech}, prob: {prob}, comm: {comm}, cult: {cult}, adapt: {adapt}")

            print(f"Debug: Running assessment for row {idx}")
            print(f"Debug: Running assessment for row {idx}")
            s_vec = np.concatenate([
                parse_skills(skills_input),
                [segment_count],
                [tech,prob,comm,cult,adapt]
            ]).reshape(1, -1) 
            print(f"Debug: Row {idx} - s_vec shape: {s_vec.shape}")
            print(f"Debug: Row {idx} - s_vec shape: {s_vec.shape}")

            j_vec = get_embedding(jd_text, dim_j)
            print(f"Debug: Row {idx} - j_vec shape: {j_vec.shape}")
            print(f"Debug: Row {idx} - j_vec shape: {j_vec.shape}")

            t_vec = get_embedding(transcript_txt, dim_t)
            print(f"Debug: Row {idx} - t_vec shape: {t_vec.shape}")
            print(f"Debug: Row {idx} - t_vec shape: {t_vec.shape}")

            feats = np.hstack([s_vec, j_vec, t_vec])
           
            print(f"Debug: Row {idx} - feats shape: {feats.shape}")
            print(f"Debug: Row {idx} - feats shape: {feats.shape}")

            if feats.shape[1] != reg_concat.n_features_in_:
                st.error(f"Row {idx} - Feature dimension mismatch: got {feats.shape[1]}, expected {reg_concat.n_features_in_}")
                print(f"Debug: Row {idx} - s_vec shape: {s_vec.shape}")
                print(f"Debug: Row {idx} - j_vec shape: {j_vec.shape}")
                print(f"Debug: Row {idx} - t_vec shape: {t_vec.shape}")
                print(f"Error: Row {idx} - Feature dimension mismatch: got {feats.shape[1]}, expected {reg_concat.n_features_in_}")
                continue

            print(f"Debug: Row {idx} - Feature dimension matches expected 808")
            print(f"Debug: Row {idx} - Feature dimension matches expected 808")
            result = run_inference(s_vec, transcript_txt, jd_text)

            # Store summary
            rec = ['Hire', 'Consider', 'Reject'][np.argmax(result['raw_probs'])]
            summary_data.append({
                "Row": idx,
                "JD ID": row['jd_id'],
                "Skills": skills_input,
                "Score": result['overall_score'],
                "Recommendation": rec,
                "Confidence": result['raw_probs'].max(),
                "Needs Review": result['needs_review']
            })

            # Detailed results in expander
            with st.expander(f"📋 Results for Row {idx}"):
                st.subheader("🔢 Scores")
                score_df = pd.DataFrame({
                    "Model": ["Concat", "Gated", "Cross-Attn", "Ensemble"],
                    "Score": [result['overall_score'], 0.0, 0.0, result['overall_score']]
                })
                st.dataframe(score_df.style.format({"Score": "{:.2f}"}))

                st.subheader("🎯 Recommendation")
                prob_df = pd.DataFrame({
                    "Recommendation": ["Hire", "Consider", "Reject"],
                    "Probability": [result['hire_prob'], result['consider_prob'], result['reject_prob']]
                })
                styled_df = prob_df.style.format({"Probability": "{:.2f}"}).map(color_recommendation, subset=["Recommendation"])
                st.dataframe(styled_df)

                if result['needs_review']:
                    st.warning("⚠️ **Low confidence** ─ Recommended for review")
                    log_for_review({
                        "jd": jd_text,
                        "transcript": transcript_txt,
                        "skills": skills_input,
                        "segment_count": segment_count,
                        "ratings": [tech, prob, comm, cult, adapt],
                        "ensemble_conf": float(result['raw_probs'].max()),
                        "ensemble_rec": rec
                    })

                tab_shap, tab_gates = st.tabs([
                    "🧠 SHAP — Top 5 Structured Features",
                    "🔀 Gate Activations"
                ])

                feats_df = pd.DataFrame(feats, columns=feature_names)

            
                with tab_shap:
                    # isolate structured features
                    struct_cols = list(skill2idx.keys()) + ["segment_count"] + [
                        "Technical_Proficiency",
                        "Problem_Solving_Ability",
                        "Communication_Skills",
                        "Cultural_Team_Fit",
                        "Adaptability_Learning",
                    ]
                    struct_df = feats_df[struct_cols]

                    explainer = shap.TreeExplainer(reg_concat)
                    shap_exp   = explainer(struct_df, check_additivity=False)
                    shap.plots.bar(shap_exp, show=False, max_display=5)
                    plt.tight_layout()
                    st.pyplot(plt)
                    plt.clf()

                with tab_gates:
                    s_t = torch.tensor(s_vec, dtype=torch.float32)
                    j_t = torch.tensor(j_vec, dtype=torch.float32)
                    t_t = torch.tensor(t_vec, dtype=torch.float32)
                    g_s = torch.sigmoid(gated.gate_s(s_t)).mean().item()
                    g_j = torch.sigmoid(gated.gate_j(j_t)).mean().item()
                    g_t = torch.sigmoid(gated.gate_t(t_t)).mean().item()
                    st.write({
                        "Struct → gate": round(g_s, 3),
                        "JD     → gate": round(g_j, 3),
                        "Text   → gate": round(g_t, 3),
                    })

            # Update progress
            progress_bar.progress((i + 1) / total_rows)

        # Display summary table
        if summary_data:
            st.header("📈 Summary of Sample Data Results")
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df.style.format({
                "Score": "{:.2f}",
                "Confidence": "{:.2f}"
            }).applymap(color_recommendation, subset=["Recommendation"]))

    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        print(f"Error: Loading sample data failed: {str(e)}")