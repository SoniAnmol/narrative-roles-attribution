# %% Import Libraries
import numpy as np
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from skopt import BayesSearchCV
from skopt.space import Integer, Real
import string
import matplotlib.pyplot as plt
import pickle


# %%
# Load and configure RoBERTa for multi-class classification
def tokenize_data(text_list, tokenizer, batch_size=8):
    return [
        tokenizer(text_list[i : i + batch_size], padding=True, truncation=True, return_tensors="pt")
        for i in range(0, len(text_list), batch_size)
    ]


def roberta_predict(texts, model, tokenizer, batch_size=8):
    batched_inputs = tokenize_data(texts, tokenizer, batch_size=batch_size)
    all_probs = []
    with torch.no_grad():
        for inputs in batched_inputs:
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).numpy()
            all_probs.extend(probs)
    return np.array(all_probs)


# Metadata preprocessing function
def preprocess_metadata(df):
    """
    Preprocess metadata for use in XGBoost.
    Includes categorical encoding and feature selection.
    """
    label_encoders = {}
    categorical_columns = ["publisher", "ARG0", "ARG1", "B-V"]

    # Encode categorical variables
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    metadata = df[["article_index", "publisher", "sentence_index", "sentiment", "ARG0", "ARG1", "B-V"]].values
    return metadata


# Metadata extraction for additional features (punctuation count)
def extract_metadata(texts):
    metadata = []
    for text in texts:
        num_punctuations = sum(1 for char in text if char in string.punctuation)
        metadata.append([num_punctuations])
    return np.array(metadata)


# Prepare data for XGBoost
def prepare_data(texts, roberta_probs, metadata):
    """
    Combine RoBERTa predictions, text features, and additional metadata.
    """
    text_features = extract_metadata(texts)
    return np.hstack([roberta_probs, text_features, metadata])


# Bayesian Optimization for XGBoost
def optimize_xgboost(X, y, num_classes_in_train):
    search_space = {
        "max_depth": Integer(3, 10),
        "learning_rate": Real(1e-3, 1e-1, prior="log-uniform"),
        "n_estimators": Integer(50, 500),
        "reg_alpha": Real(1e-5, 1e-2, prior="log-uniform"),
        "reg_lambda": Real(1e-5, 1e-2, prior="log-uniform"),
        "subsample": Real(0.5, 1.0),
        "min_child_weight": Integer(1, 10),
        "gamma": Real(0, 5),
    }

    xgb = XGBClassifier(
        objective="multi:softmax", num_class=num_classes_in_train, eval_metric="mlogloss", tree_method="hist"
    )

    optimizer = BayesSearchCV(
        estimator=xgb, search_spaces=search_space, n_iter=30, cv=3, n_jobs=-1, verbose=1, scoring="f1_weighted"
    )

    optimizer.fit(X, y)
    return optimizer.best_estimator_


# One-Hot Encoding of Predictions
def get_one_hot_encoded_predictions(y_pred, num_classes):
    """
    Convert predicted labels to one-hot encoded format.
    """
    one_hot_encoder = OneHotEncoder(categories="auto")
    y_pred_reshaped = y_pred.reshape(-1, 1)
    one_hot_encoder.fit(np.arange(num_classes).reshape(-1, 1))
    return one_hot_encoder.transform(y_pred_reshaped)


# Evaluate metrics per class
def evaluate_per_class(y_true, y_pred, label_encoder, top_n=10, output_file="../output/prediction_performance.xlsx"):
    """
    Evaluate metrics (F1 score, accuracy, and recall) for each class and plot the top N classes.

    Args:
        y_true (array): True labels (encoded).
        y_pred (array): Predicted labels (encoded).
        label_encoder (LabelEncoder): Fitted LabelEncoder for decoding class labels.
        top_n (int): Number of top classes to plot based on F1 score.
    """

    unique_classes = label_encoder.classes_
    class_metrics = {}

    for cls in unique_classes:
        cls_index = label_encoder.transform([cls])[0]
        f1 = f1_score(y_true == cls_index, y_pred == cls_index, average="binary")
        cls_precision = precision_score(y_true[y_true == cls_index], y_pred[y_true == cls_index], average="binary")
        cls_recall = recall_score(y_true == cls_index, y_pred == cls_index, average="binary")
        class_metrics[cls] = {"f1_score": f1, "precision": cls_precision, "recall": cls_recall}

    # Convert metrics dictionary to DataFrame
    metrics_df = pd.DataFrame.from_dict(class_metrics, orient="index").reset_index()
    metrics_df.columns = ["class", "f1_score", "precision", "recall"]

    # Save metrics to an Excel file
    metrics_df.to_excel(output_file, index=False)
    print(f"Metrics exported to {output_file}")

    # Sort by F1 score to get top N classes
    class_metrics.pop("nan", None)
    sorted_metrics = sorted(class_metrics.items(), key=lambda x: x[1]["f1_score"], reverse=True)[:top_n]
    top_classes, top_scores = zip(*sorted_metrics)

    # Extract metrics
    f1_scores = [scores["f1_score"] for scores in top_scores]
    precisions = [scores["precision"] for scores in top_scores]
    recalls = [scores["recall"] for scores in top_scores]

    # Plotting
    x = np.arange(len(top_classes))
    bar_width = 0.4

    plt.figure(figsize=(12, 8))
    plt.barh(x, f1_scores, bar_width, label="F1 Score", color="skyblue")

    # Scatter plots for accuracy and recall
    plt.scatter(precisions, x, color="orange", marker="x", label="Precisions", s=100)
    plt.scatter(recalls, x, color="orange", marker="d", label="Recall", s=100)

    # Add a vertical threshold line at 0.6
    plt.axvline(x=0.6, color="red", linestyle="--", label="Threshold")

    # Formatting the plot
    plt.yticks(x, top_classes)
    plt.xlabel("Score")
    plt.ylabel("Classes")
    plt.title(f"Top {top_n} F1 Score")
    plt.legend(loc="lower right")
    plt.gca().invert_yaxis()  # For readability, highest score on top
    plt.savefig("../figures/model_performance.png", transparent=True)
    plt.show()


def prepare_annotated_data(df):
    """
    Maps values in 'arg0_role' and 'arg1_role', creates a new column 'character_role',
    and appends rows with non-null 'character_role_arg1' to the original DataFrame,
    keeping only original columns and 'character_role'.

    Args:
    df (pd.DataFrame): DataFrame containing 'ARG0', 'arg0_role', 'ARG1', and 'arg1_role' columns.

    Returns:
    pd.DataFrame: Original DataFrame with appended rows and only the original columns + 'character_role'.
    """
    # Define the mapping
    role_mapping = {0: None, 1: "hero", 2: "villain", 3: "victim"}

    # Map and concatenate for arg0_role
    df["mapped_role_arg0"] = df["arg0_role"].map(role_mapping)
    df["character_role_arg0"] = df.apply(
        lambda row: f"{row['ARG0']}-{row['mapped_role_arg0']}" if pd.notnull(row["mapped_role_arg0"]) else None, axis=1
    )

    # Map and concatenate for arg1_role
    df["mapped_role_arg1"] = df["arg1_role"].map(role_mapping)
    df["character_role_arg1"] = df.apply(
        lambda row: f"{row['ARG1']}-{row['mapped_role_arg1']}" if pd.notnull(row["mapped_role_arg1"]) else None, axis=1
    )

    # Create a combined 'character_role' column using non-null 'character_role_arg1' first
    df["character_role"] = df["character_role_arg1"].combine_first(df["character_role_arg0"])

    # Filter rows with non-null 'character_role_arg1'
    non_null_arg1_roles = df[df["character_role_arg1"].notna()].copy()

    # Append the filtered rows to the original DataFrame
    result_df = pd.concat([df, non_null_arg1_roles]).reset_index(drop=True)

    # Retain only original columns and 'character_role'
    drop_columns = [
        "mapped_role_arg0",
        "character_role_arg0",
        "mapped_role_arg1",
        "character_role_arg1",
    ]
    result_df.drop(columns=drop_columns, inplace=True)
    return result_df


# %%
# Main workflow
if __name__ == "__main__":
    # Load annotated dataset
    training_data = pd.read_excel("../output/annotated_data.xlsx")
    actor_directory = pd.read_csv("../data/actor_entity_directory_v2.csv")
    unique_roles = pd.read_csv("../data/unique_actor_character_roles.csv")

    # %%
    # * prepare training dataset
    actor_directory = actor_directory[actor_directory.actor.notna()].reset_index(drop=True)
    actor_directory = dict(zip(actor_directory["category"], actor_directory["actor"]))

    # * label actors with standard actor names
    training_data["ARG0"] = training_data["ARG0"].map(actor_directory).fillna(training_data["ARG0"])
    training_data["ARG1"] = training_data["ARG1"].map(actor_directory).fillna(training_data["ARG1"])

    # * Prepare annotated data
    data = prepare_annotated_data(training_data)

    # %%
    # Define the list of valid roles
    unique_roles = unique_roles["character-role"].to_list()

    # %%
    # Check if 'character_role' is in the list_roles, if not, assign NaN
    data["character_role"] = data["character_role"].apply(lambda x: x if x in unique_roles else np.nan)

    # %%
    # Step 1: Define a threshold
    threshold = 3  # Minimum number of samples required per class

    # Step 2: Identify rare classes
    class_counts = data["character_role"].value_counts()
    rare_classes = class_counts[class_counts < threshold].index

    # Step 3: Remove rows with rare classes
    data = data[~data["character_role"].isin(rare_classes)]

    # %%
    texts = data["sentence"].tolist()
    y = data["character_role"].tolist()

    # %%
    # Label encode character_role
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # %%
    # Prepare metadata
    metadata = preprocess_metadata(data)

    # %%
    # Load and configure RoBERTa
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    num_classes = len(label_encoder.classes_)
    model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=num_classes)

    # %%
    # Get RoBERTa predictions
    roberta_probs = roberta_predict(texts, model, tokenizer)

    # %%
    # Prepare features
    X = prepare_data(texts, roberta_probs, metadata)

    # %%
    # Split dataset
    X_train, X_test, y_train, y_test, data_train, data_test = train_test_split(
        X, y_encoded, data, test_size=0.2, random_state=42
    )

    # %%
    # No need to refit LabelEncoder, directly use transformed y_train and y_test
    num_classes = len(np.unique(y_train))  # Ensure correct num_classes for XGBoost

    # %%
    # Train and optimize XGBoost
    best_xgb = optimize_xgboost(X_train, y_train, num_classes)

    # %%
    # Make predictions
    y_pred = best_xgb.predict(X_test)
    # y_pred_one_hot = get_one_hot_encoded_predictions(y_pred, num_classes)

    # %%
    # General metrics
    print("Overall Accuracy:", accuracy_score(y_test, y_pred))
    print("Overall F1 Score (weighted):", f1_score(y_test, y_pred, average="weighted"))

    # %%
    # Assuming y_pred contains the encoded predictions
    y_pred_original = label_encoder.inverse_transform(y_pred)

    # Print some decoded predictions
    print("Decoded Predictions (Original Labels):", y_pred_original[:5])

    # %%
    evaluate_per_class(y_test, y_pred, label_encoder, top_n=15)

    # %%
    # * Save trained model
    with open("best_xgb_model.pkl", "wb") as f:
        pickle.dump(best_xgb, f)

    # %%
    ##########################################################################-#-
    data = pd.read_csv("../output/news_narratives_2024-11-05_20-50-54.csv.gz")

    # %%
    # Replace values in the specified columns
    data["ARG0"] = data["ARG0"].map(actor_directory).fillna(data["ARG0"])
    data["ARG1"] = data["ARG1"].map(actor_directory).fillna(data["ARG1"])

    # %%
    texts = data["sentence"].tolist()
    metadata = preprocess_metadata(data)

    # %%
    # Get RoBERTa predictions
    roberta_probs = roberta_predict(texts, model, tokenizer)

    # %%
    # Prepare features
    X = prepare_data(texts, roberta_probs, metadata)

    # %%
    # Make predictions
    y_pred = best_xgb.predict(X)
    y_pred_one_hot = get_one_hot_encoded_predictions(y_pred, num_classes)

    # %%
    # Assuming y_pred contains the encoded predictions
    y_pred_original = label_encoder.inverse_transform(y_pred)

    # %%
    data = pd.read_csv("../output/news_narratives_2024-11-05_20-50-54.csv.gz")
    data["predicted_character_role"] = y_pred_original
    data["predicted_character_role"] = data["predicted_character_role"].replace("nan", np.nan)
    # data = data.dropna(subset="predicted_character_role")
    # Save or inspect results
    data.to_csv("predicted_results.csv.gz", index=False, compression="gzip")
    # %%
