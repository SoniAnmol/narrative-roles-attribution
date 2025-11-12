"""This script contains code for preparing the lexis-nexis data downloaded from the database."""

# %%
import os
import re
import pandas as pd
from docx import Document


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
    with open(filepath, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Skip initial blank lines
    lines = [line.strip() for line in lines if line.strip()]

    # Initialize extracted values
    title, publisher, section, length, byline, highlight, body, load_date = (None,) * 8

    # Extract title and publisher
    title = lines[0].strip() if len(lines) > 0 else None
    publisher = lines[1].strip() if len(lines) > 1 else None

    # Regex patterns for each field
    section_pattern = re.compile(r"^Section:\s*(.*)")
    length_pattern = re.compile(r"^Length:\s*(\d+)\s*words")
    byline_pattern = re.compile(r"^Byline:\s*(.*)")
    highlight_pattern = re.compile(r"^Highlight:\s*(.*)")
    load_date_pattern = re.compile(r"^Load-Date:\s*(.*)")

    # Flags and content gathering
    body_started = False
    body_lines = []

    # Loop through the lines to capture other sections
    for line in lines[2:]:
        line = line.strip()

        if section is None and section_pattern.match(line):
            section = section_pattern.match(line).group(1)

        elif length is None and length_pattern.match(line):
            length = length_pattern.match(line).group(1)

        elif byline is None and byline_pattern.match(line):
            byline = byline_pattern.match(line).group(1)

        elif highlight is None and highlight_pattern.match(line):
            highlight = highlight_pattern.match(line).group(1)

        elif line == "Body":
            body_started = True

        elif body_started and not load_date_pattern.match(line):
            body_lines.append(line)

        elif load_date_pattern.match(line):
            load_date = load_date_pattern.match(line).group(1)
            body_started = False  # Stop body collection at load date

    body = " ".join(body_lines).strip() if body_lines else None

    # Return a dictionary for DataFrame creation
    return {
        "Title": title,
        "Publisher": publisher,
        "Section": section,
        "Word Count": length,
        "Byline": byline,
        "Highlight": highlight,
        "Body": body,
        "Load Date": load_date,
    }


# %%
folder_path = "data/lexis_nexis"
# %%
# Set the folder path containing .docx files
docx_to_txt(folder_path)

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
# %%
# Create DataFrame from extracted data
df = pd.DataFrame(data)

# Display the DataFrame
print(df)
# %%
# Optionally save to CSV
df.to_csv(f"{folder_path}/ln_articles.csv.gz", index=False, compression="gzip")
