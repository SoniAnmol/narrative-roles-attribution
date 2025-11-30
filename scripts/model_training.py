"""
This script trains a RoBERTa + XGBoost hybrid model to classify
sentences into narrative character roles.

Includes:
- Rare label dropping (no merging)
- Global class weighting for imbalance
- Metadata integration (publisher, ARG0, ARG1, date, sentiment)
- Modular training and prediction architecture
- Improved evaluation reporting
- Easily switch between 'roberta-base' and 'roberta-large'
"""

#%% -------------------- IMPORTS --------------------
import pandas as pd
import numpy as np
import random
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from transformers import RobertaTokenizer, RobertaModel
import torch
from tqdm import tqdm
import joblib

#%% ------------------- GLOBAL CONFIG ------------------
SEED = 42
MODEL_NAME = "roberta-large" 

# slightly longer max_length for large, same code for both
MAX_LEN = 192 if "large" in MODEL_NAME else 128
BATCH_SIZE = 16  # used for batched embedding

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.use_deterministic_algorithms(True)

#%%  ROLE PREPARATION
def prepare_annotated_data(df, actor_roles):
    # code actor roles in the training data
    role_dict = {0: 'none',
        1: 'hero',
        2: 'villain',
        3: 'victim'}

    # map training data with roles
    for col in ['arg0_role', 'arg1_role']:
        df[col] = df[col].map(role_dict).astype(str)
        actor = col.split("_")[0].upper()
        df[col] = df[actor] + "-" + df[col]
        df.loc[~df[col].isin(actor_roles), col] = "none"
    
    # create a training matrix using one-hot encoding from columns 'arg0_role' and 'arg1_role'
    all_roles = actor_roles + ["none"]

    # Initialize all one-hot columns to 0
    for role in all_roles:
        df[role] = 0

    # For each row, activate the relevant columns
    for idx, row in df.iterrows():
        roles = {row["arg0_role"], row["arg1_role"]}
        for r in roles:
            if r in all_roles:
                df.at[idx, r] = 1

    df["sentence_word_count"] = (
        df["sentence"]
        .astype(str)
        .str.strip()
        .str.split()
        .apply(len)
    )

    df[all_roles] = df[all_roles].astype('int8')
    df.drop(columns=["none"], inplace=True)
    return df

# ROLE COUNTING
def count_actor_roles(df, actor_roles, output_path=None):
    rows = []
    for col in actor_roles:
        if col not in df.columns:
            continue
        actor, role = col.split("-")
        rows.append({"Actor": actor, "Role": role, "Count": int(df[col].sum())})
    role_df = pd.DataFrame(rows).sort_values(["Actor", "Role"])
    if output_path:
        role_df.to_excel(output_path, index=False)
    return role_df

# ROBERTA LOADER (works for base & large)
def load_roberta(model_name: str):
    """
    Load RoBERTa tokenizer and model for the given model_name.
    Automatically moves model to CPU/GPU and returns device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device

# EMBEDDINGS (batched, model-agnostic)
def embed_text(text_list, tokenizer, model, device, max_length=128, batch_size=16):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size), desc=f"Embedding with {MODEL_NAME}"):
            batch = text_list[i:i + batch_size]
            inputs = tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            outputs = model(**inputs)
            cls_batch = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_batch)
    return np.vstack(embeddings)

# STRUCTURED FEATURES
def build_structured_features(df, ohe=None, fit=False):
    """
    Convert metadata into numerical features.
    If fit=True, fits the OneHotEncoder; else uses existing encoder.
    """
    feat_df = pd.DataFrame(index=df.index)

    # ----------------------------------------------------
    # Numeric features
    # ----------------------------------------------------
    feat_df["sentiment"] = df.get("sentiment", 0).astype(float)

    # Add sentence_word_count safely
    feat_df["sentence_word_count"] = df.get("sentence_word_count", 0).astype(float)

    # Date features
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        feat_df["year"] = dt.dt.year.fillna(0)
        feat_df["month"] = dt.dt.month.fillna(0)
    else:
        feat_df["year"] = 0
        feat_df["month"] = 0

    # ----------------------------------------------------
    # Categorical features
    # ----------------------------------------------------
    cat_cols = [c for c in ["publisher", "ARG0", "ARG1"] if c in df.columns]

    if cat_cols:
        if ohe is None and fit:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            ohe_matrix = ohe.fit_transform(df[cat_cols])
        else:
            ohe_matrix = ohe.transform(df[cat_cols])

        ohe_cols = ohe.get_feature_names_out(cat_cols)

        feat_df = pd.concat(
            [feat_df,
             pd.DataFrame(ohe_matrix, columns=ohe_cols, index=df.index)],
            axis=1
        )

    return feat_df, ohe

def combine_features(embeddings, struct_features):
    struct_np = struct_features.to_numpy().astype(np.float32)
    return np.hstack([embeddings, struct_np])

# LABEL FILTERING 
def filter_rare_labels(df, label_cols, min_pos=10):
    """Drop labels with fewer than min_pos positive examples."""
    label_sums = df[label_cols].sum()
    valid = label_sums[label_sums >= min_pos].index.tolist()
    return valid

# CLASS WEIGHTS 
def compute_global_weight(y_train):
    """
    Compute a single global scale_pos_weight for XGBoost
    based on all labels combined.
    """
    y_values = y_train.values
    total_pos = y_values.sum()
    total_neg = y_values.size - total_pos

    if total_pos == 0:
        return 1.0

    return float(total_neg / total_pos)

# TRAINING MODULE
def train_model(X_train_final, y_train, global_weight):
    """
    Train a MultiOutputClassifier with a single XGBClassifier
    using a global class weight for imbalance.
    """

    base_estimator = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=global_weight,
        random_state=SEED,
    )

    clf = MultiOutputClassifier(base_estimator)
    clf.fit(X_train_final, y_train)
    return clf

# PREDICTION MODULE 
def predict_binary(clf, X_test_final, threshold=0.5):
    """Predict with adjustable threshold."""
    probas = clf.predict_proba(X_test_final)
    preds = np.zeros((X_test_final.shape[0], len(probas)), dtype=int)

    for i, p in enumerate(probas):
        preds[:, i] = (p[:, 1] > threshold).astype(int)

    return preds

#%% ---------------------- MAIN -----------------------
if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parent.parent
    annotated_data = pd.read_excel(
        f"{ROOT}/data/training_data/annotated_training_data.xlsx"
    )

    actor_roles = [
        "EU-hero",
        "agriculture-hero",
        "business-hero",
        "civil society-hero",
        "emergency service-hero",
        "environment-hero",
        "essential goods and infrastructure-hero",
        "municipality-hero",
        "national government-hero",
        "people-hero",
        "political actors-hero",
        "region-hero",
        "river monitoring agency-hero",
        "EU-villain",
        "agriculture-villain",
        "business-villain",
        "civil society-villain",
        "climate change-villain",
        "emergency service-villain",
        "environment-villain",
        "essential goods and infrastructure-villain",
        "extreme event-villain",
        "municipality-villain",
        "national government-villain",
        "people-villain",
        "political actors-villain",
        "region-villain",
        "river monitoring agency-villain",
        "agriculture-victim",
        "business-victim",
        "emergency service-victim",
        "environment-victim",
        "essential goods and infrastructure-victim",
        "municipality-victim",
        "national government-victim",
        "people-victim",
        "political actors-victim",
        "region-victim",
    ]
    
    # Prepare roles
    annotated_data = prepare_annotated_data(annotated_data, actor_roles)
    count_actor_roles(annotated_data, actor_roles)

    TEXT_COL = "sentence"
    annotated_data[TEXT_COL] = annotated_data[TEXT_COL].astype(str)

    # Remove labels with no variation
    y = annotated_data[actor_roles]
    non_constant = y.loc[:, y.nunique() > 1].columns.tolist()

    # Remove labels with too few positives
    label_cols = filter_rare_labels(annotated_data, non_constant, min_pos=5)

    print(f"Using {len(label_cols)} labels:", label_cols)

    # Keep structured metadata
    struct_cols = ["publisher", "ARG0", "ARG1", "sentiment", "date", "sentence_word_count"]
    struct_available = [c for c in struct_cols if c in annotated_data.columns]

    # SPLIT
    X_train_text, X_test_text, y_train, y_test, meta_train, meta_test = train_test_split(
        annotated_data[TEXT_COL],
        annotated_data[label_cols],
        annotated_data[struct_available],
        test_size=0.20,
        random_state=SEED,
        stratify=(annotated_data[label_cols].sum(axis=1) > 0),
    )

    # Load RoBERTa (base or large, depending on MODEL_NAME)
    tokenizer, model, device = load_roberta(MODEL_NAME)
    print(f"Loaded {MODEL_NAME} on device: {device}")

    # Embed
    X_train_emb = embed_text(
        X_train_text.tolist(),
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=MAX_LEN,
        batch_size=BATCH_SIZE,
    )
    X_test_emb = embed_text(
        X_test_text.tolist(),
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=MAX_LEN,
        batch_size=BATCH_SIZE,
    )

    # Fit encoder on train metadata
    X_train_struct, ohe = build_structured_features(meta_train, ohe=None, fit=True)
    # Transform test metadata using SAME encoder
    X_test_struct, _ = build_structured_features(meta_test, ohe=ohe, fit=False)

    X_train_final = combine_features(X_train_emb, X_train_struct)
    X_test_final  = combine_features(X_test_emb,  X_test_struct)

    # Compute class weights
    global_weight = compute_global_weight(y_train)
    print(f"Using global scale_pos_weight = {global_weight:.2f}")

    clf = train_model(X_train_final, y_train, global_weight)

    # Save model

    joblib.dump(clf, f"{ROOT}/models/classifier/roberta_xgb_model.pkl")
    joblib.dump(label_cols, f"{ROOT}/models/classifier/label_cols.pkl")
    joblib.dump(struct_available, f"{ROOT}/models/classifier/struct_feature_cols.pkl")
    joblib.dump(ohe, f"{ROOT}/models/classifier/ohe_encoder.pkl")
    joblib.dump(X_train_struct.columns.tolist(), f"{ROOT}/models/classifier/ohe_struct_feature_names.pkl")
    joblib.dump(
        {"model_name": MODEL_NAME, "max_len": MAX_LEN, "batch_size": BATCH_SIZE},
        f"{ROOT}/models/classifier/text_encoder_config.pkl",
    )

    # Predict
    y_pred = predict_binary(clf, X_test_final, threshold=0.50)

    # Better evaluation
    print("=== CLASSIFICATION REPORT ===")
    report_dict = classification_report(
    y_test,
    pd.DataFrame(y_pred, columns=label_cols),
    target_names=label_cols,
    zero_division=0,
    output_dict=True)
    #%%
    # Convert to DataFrame
    report_df = pd.DataFrame(report_dict).transpose()
    # Export to Excel
    report_df.to_excel(f"{ROOT}/data/model_performance/classification_report.xlsx", index=True)

    print(report_df)
# %%
