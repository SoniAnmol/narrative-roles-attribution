"""This script contains code for preparing the lexis-nexis data downloaded from the database."""

# %%
import os
import re
import pandas as pd
from docx import Document
from pathlib import Path
from unidecode import unidecode

#%%
MONTH_MAP_IT = {
    "Gennaio": "January", "Febbraio": "February", "Marzo": "March",
    "Aprile": "April", "Maggio": "May", "Giugno": "June",
    "Luglio": "July", "Agosto": "August", "Settembre": "September",
    "Ottobre": "October", "Novembre": "November", "Dicembre": "December",
}

WEEKDAYS_IT = [
    "Lunedì", "Martedì", "Mercoledì",
    "Giovedì", "Venerdì", "Sabato", "Domenica"
]

def clean_lexis_date(raw):
    """Normalize English + Italian LexisNexis date formats."""
    if not raw or not isinstance(raw, str):
        return None

    # Remove weird spaces
    raw = raw.replace("\xa0", " ").replace("\u200b", "").strip()

    # Remove Italian weekday at end
    for wd in WEEKDAYS_IT:
        raw = re.sub(rf"\s*{wd}$", "", raw, flags=re.IGNORECASE).strip()

    # Replace Italian month with English equivalent
    for it, en in MONTH_MAP_IT.items():
        raw = re.sub(it, en, raw, flags=re.IGNORECASE)

    # Remove accents (Venerdì → Venerdi)
    raw = unidecode(raw)

    return raw.strip()

def robust_parse_date(date_str):
    if pd.isna(date_str):
        return pd.NaT

    # Normalize unicode
    s = (
        str(date_str)
        .replace("\xa0", " ")
        .replace("\u200b", "")
        .strip()
    )

    # Remove weekday if still present
    # "24 August 2024 Saturday" → "24 August 2024"
    s = re.sub(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)$", "", s, flags=re.IGNORECASE).strip()

    # Try direct pandas parse
    dt = pd.to_datetime(s, errors="ignore")

    if isinstance(dt, pd.Timestamp):
        return dt

    # Fallback: tell pandas dayfirst format
    try:
        return pd.to_datetime(s, dayfirst=True)
    except:
        return pd.NaT


def docx_to_txt(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    # Gather all .docx files in the folder
    docx_files = [f for f in os.listdir(folder_path) if f.endswith(".DOCX")]

    # Check if there are any .docx files in the folder
    if not docx_files:
        print("No .docx files found in the folder.")
        return

    # Loop through all .docx files and convert them to .txt
    for filename in docx_files:
        docx_path = os.path.join(folder_path, filename)
        txt_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.txt")

        # Read the .docx file
        doc = Document(docx_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])

        # Write the text to a .txt file
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(text)

        print(f"Converted '{filename}' to '{os.path.basename(txt_path)}'.")

# Function to extract data from each file
def extract_article_data(filepath):
    filename = os.path.basename(filepath)

    # Regex to detect Italian style dates like:
    # "19 Settembre 2024 Giovedì"
    italian_date_pattern = re.compile(
        r"^\d{1,2}\s+[A-Za-zÀ-ÿ]+\s+\d{4}(?:\s+[A-Za-zÀ-ÿ]+)?$"
    )

    # Read file
    with open(filepath, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Remove blank lines
    lines = [line.strip() for line in lines if line.strip()]

    # Initialize fields
    title = publisher = section = length = byline = highlight = None
    load_date_raw = None
    body = None

    # Regex patterns for metadata
    section_pattern = re.compile(r"^Section:\s*(.*)")
    length_pattern = re.compile(r"^Length:\s*(\d+)\s*words")
    byline_pattern = re.compile(r"^Byline:\s*(.*)")
    highlight_pattern = re.compile(r"^Highlight:\s*(.*)")
    load_date_pattern = re.compile(r"^Load-Date:\s*(.*)")

    # First lines: title + publisher
    if len(lines) > 0:
        title = lines[0]
    if len(lines) > 1:
        publisher = lines[1]

    body_started = False
    body_lines = []

    # Process the rest of the file
    for line in lines[2:]:
        # 1. Standard Lexis Nexis format: "Load-Date: June 15, 2023"
        if m := load_date_pattern.match(line):
            load_date_raw = m.group(1).strip()
            body_started = False
            continue

        # 2. Italian WebNews format: "19 Settembre 2024 Giovedì"
        if load_date_raw is None and italian_date_pattern.match(line):
            load_date_raw = line.strip()
            continue

        # Section
        if m := section_pattern.match(line):
            section = m.group(1)
            continue

        # Length
        if m := length_pattern.match(line):
            length = m.group(1)
            continue

        # Byline
        if m := byline_pattern.match(line):
            byline = m.group(1)
            continue

        # Highlight
        if m := highlight_pattern.match(line):
            highlight = m.group(1)
            continue

        # Start of article body
        if line == "Body":
            body_started = True
            continue

        # Collect body text
        if body_started:
            body_lines.append(line)

    # Finalize body
    body = " ".join(body_lines).strip() if body_lines else None

    # Clean the extracted date (Italian → English)
    load_date_clean = clean_lexis_date(load_date_raw)

    return {
        "filename": filename,
        "Title": title,
        "Publisher": publisher,
        "Section": section,
        "Word Count": length,
        "Byline": byline,
        "Highlight": highlight,
        "Body": body,
        "Load Date Raw": load_date_raw,
        "Load Date Clean": load_date_clean,
    }



# %%
ROOT = Path(__file__).resolve().parent.parent 
folder_path = ROOT / "data/lexis_nexis"
# %%
# Set the folder path containing .docx files
# docx_to_txt(folder_path)

# %%
# Initialize a list to collect article data
data = []

# Loop over all text files in the directory
for filename in os.listdir(folder_path):
    filepath = os.path.join(folder_path, filename)
    if filename.endswith(".txt"):
        try:
            article_data = extract_article_data(filepath)
            data.append(article_data)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
# %% Create DataFrame from extracted data
df = pd.DataFrame(data)

#%% Parse and clean Load Date

df["date"] = df["Load Date Clean"].apply(robust_parse_date)

#%%
df = df[df.date.notna()].drop(columns=['Load Date Raw', 'Load Date Clean']).rename(columns={'date':'Load Date'})

# %%
# Optionally save to CSV
df.to_csv(f"{folder_path}/ln_articles.csv.gz", index=False, compression="gzip")

# %%
