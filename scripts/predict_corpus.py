#%% import libraries
import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import model_training as mt
from model_training import build_structured_features, embed_text
import gc
#%% Global Config 
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)

#%% Load models and config
ROOT = Path(__file__).resolve().parent.parent

clf = joblib.load(f"{ROOT}/models/classifier/roberta_xgb_model.pkl")
label_cols = joblib.load(f"{ROOT}/models/classifier/label_cols.pkl")
struct_feature_cols = joblib.load(f"{ROOT}/models/classifier/struct_feature_cols.pkl")
ohe = joblib.load(f"{ROOT}/models/classifier/ohe_encoder.pkl")
ohe_struct_names = joblib.load(f"{ROOT}/models/classifier/ohe_struct_feature_names.pkl")
text_config = joblib.load(f"{ROOT}/models/classifier/text_encoder_config.pkl")

MODEL_NAME = text_config["model_name"]
MAX_LEN = text_config["max_len"]
BATCH_SIZE = text_config["batch_size"]

# override globals inside model_training
mt.MODEL_NAME = MODEL_NAME
mt.MAX_LEN = MAX_LEN
mt.BATCH_SIZE = BATCH_SIZE

#%% Load RoBERTa
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
roberta_model = RobertaModel.from_pretrained(MODEL_NAME).to(device).eval()

#%% Load full corpus
df_corpus = pd.read_csv(f"{ROOT}/data/news_corpus/news_corpus_svo.csv.gz", compression='gzip')
df_corpus.rename(columns={"sentences":"sentence"}, inplace=True)
df_corpus["sentence"] = df_corpus["sentence"].astype(str)

#%% Build embeddings and features
CHUNK_SIZE = 2000   # recommended starting value

results_dir = ROOT / "data/predictions/chunks"
results_dir.mkdir(parents=True, exist_ok=True)

for start in range(0, len(df_corpus), CHUNK_SIZE):
    end = min(start + CHUNK_SIZE, len(df_corpus))
    print(f"Processing rows {start}–{end}")

    df_chunk = df_corpus.iloc[start:end].copy()

    # Add word count
    df_chunk["sentence_word_count"] = (
        df_chunk["sentence"].str.strip().str.split().apply(len)
    )

    # GPU embeddings
    X_text = df_chunk["sentence"].tolist()
    X_emb = embed_text(
        text_list=X_text,
        tokenizer=tokenizer,
        model=roberta_model,
        device=device,
        max_length=MAX_LEN,
        batch_size=BATCH_SIZE
    )

    # Structured features
    X_struct, _ = build_structured_features(df_chunk, ohe=ohe, fit=False)

    X_final = np.hstack([X_emb, X_struct.to_numpy().astype(np.float32)])

    # Predict
    probas = clf.predict_proba(X_final)

    # Store chunk predictions
    binary_preds = np.zeros((X_final.shape[0], len(label_cols)), dtype=int)
    prob_df = pd.DataFrame(index=df_chunk.index)

    for i, lbl in enumerate(label_cols):
        prob_df[lbl + "_prob"] = probas[i][:, 1]
        binary_preds[:, i] = (probas[i][:, 1] > 0.5).astype(int)

    pred_df = pd.DataFrame(binary_preds, columns=label_cols, index=df_chunk.index)

    out_chunk = pd.concat([df_chunk, pred_df, prob_df], axis=1)

    out_chunk.to_parquet(results_dir / f"pred_{start}_{end}.parquet")

    # FREE MEMORY
    del X_emb, X_struct, X_final, probas, pred_df, out_chunk
    torch.cuda.empty_cache()
    gc.collect()

#%% Combine predictions with original corpus
chunk_files = sorted(results_dir.glob("pred_*.parquet"))
output = pd.concat([pd.read_parquet(cf) for cf in chunk_files], ignore_index=True)


#%% Save results
out_path = f"{ROOT}/data/predictions/corpus_predictions.csv.gz"
output.to_csv(out_path, index=False, compression='gzip')
print(f"Saved predictions to {out_path}")
print(output.head())

# %%
