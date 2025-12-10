"""This script reads the newspaper data and prepare it for analysis"""

# %%
# import libraries
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from deep_translator import GoogleTranslator
from langdetect import detect
import nltk
import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
# nltk.download('punkt')
#%%

def detect_language(text):
    """
    Detect the language of the given text.

    :param text: The text to detect the language for.
    :return: The detected language code or 'error' if detection fails.
    """
    try:
        # Detect the language
        language = detect(text)
        return language
    except Exception:
        return "error"

def plot_language_distribution(df):
    """
    Plot the distribution of detected languages in a pie chart.

    :param df: DataFrame with the detected language information.
    """
    # Plot the language distribution
    language_counts = df["language"].value_counts()
    plt.figure(figsize=(8, 8))
    language_counts.plot(kind="pie", autopct="%1.1f%%", startangle=140)
    plt.title("Distribution of Detected Languages")
    plt.ylabel("")  # Hide the y-label
    plt.show()

def split_text_into_chunks(text, max_length=5000):
    """
    Split the text into chunks with a maximum length, ensuring that chunks
    are created at sentence boundaries.

    Parameters:
    - text (str): Input text.
    - max_length (int): Maximum length of each chunk.

    Returns:
    - List[str]: List of text chunks.
    """
    sentences = nltk.tokenize.sent_tokenize(text, language="italian")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding the next sentence exceeds the max_length, save the current chunk and start a new one
        if len(current_chunk) + len(sentence) + 1 > max_length:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            # Add the sentence to the current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def translate(text, source_lang=None, target_lang="en"):
    """
    Translate text to English.

    Parameters:
    - text (str): Input text.
    - source_lang (str, optional): Source language code. If None, it will be detected.
    - target_lang (str): Target language code.

    Returns:
    - str: Translated text in English or original text if translation fails.
    """
    if source_lang is None:
        source_lang = detect_language(text)
    if source_lang == "error":
        return text  # Return original text if language detection fails

    try:
        # Split text into chunks
        chunks = split_text_into_chunks(text)
        translated_chunks = [
            GoogleTranslator(source=source_lang, target=target_lang).translate(chunk) for chunk in chunks
        ]
        return " ".join(translated_chunks)
    except Exception as e:
        return text  # Return original text if translation fails


# def remove_similar_texts(df, text_column="text", date_column="date", similarity_threshold=0.90):
    # """
    # Remove duplicate or highly similar texts based on embedding similarity and date matching.
    # If date is NaN, drops duplicate texts based on the text column only.

    # Parameters:
    #     df (pd.DataFrame): DataFrame containing the data.
    #     text_column (str): The name of the column with text data.
    #     date_column (str): The name of the column with date data.
    #     similarity_threshold (float): Threshold for cosine similarity to consider texts as duplicates.

    # Returns:
    #     pd.DataFrame: Cleaned DataFrame with similar texts removed.
    # """
    # # Load the embedding model
    # model = SentenceTransformer("all-MiniLM-L6-v2")

    # # Generate embeddings for all text entries
    # embeddings = model.encode(df[text_column].tolist(), convert_to_tensor=True)

    # # Calculate pairwise similarity matrix
    # similarity_matrix = util.cos_sim(embeddings, embeddings).numpy()

    # # Set to keep track of indices to drop
    # to_drop = set()

    # # Loop through the similarity matrix to find duplicates
    # for i in range(len(df)):
    #     for j in range(i + 1, len(df)):
    #         # Check if similarity exceeds threshold
    #         if similarity_matrix[i][j] >= similarity_threshold:
    #             # Check if the date is NaN for both entries
    #             if pd.isna(df[date_column][i]) and pd.isna(df[date_column][j]):
    #                 # Mark duplicate based on text only, keeping the first occurrence
    #                 if j not in to_drop:
    #                     to_drop.add(j)
    #             elif df[date_column][i] == df[date_column][j]:
    #                 # Compare word counts and mark the shorter text for dropping
    #                 if len(df[text_column][i].split()) < len(df[text_column][j].split()):
    #                     to_drop.add(i)
    #                 else:
    #                     to_drop.add(j)

    # # Drop entries with identified indices
    # df_cleaned = df.drop(to_drop).reset_index(drop=True)

    # return df_cleaned

def remove_similar_texts(
    df, 
    text_column="text", 
    date_column="date",
    similarity_threshold=0.90,
):
    """
    Remove near-duplicate texts using embedding similarity clustering.
    Keeps ONE representative per cluster (the longest text or earliest date).
    """

    # Clean index
    df = df.reset_index(drop=True)

    # Load model once
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = df[text_column].astype(str).tolist()

    # Compute embeddings
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)

    # Compute similarity matrix
    sim = util.cos_sim(embeddings, embeddings).cpu().numpy()

    n = len(df)
    visited = np.zeros(n, dtype=bool)
    keep_indices = []

    for i in range(n):
        if visited[i]:
            continue

        # start cluster with element i
        cluster = [i]
        visited[i] = True

        # find all items similar to i
        for j in range(i + 1, n):
            if sim[i, j] >= similarity_threshold:
                cluster.append(j)
                visited[j] = True

        # pick representative from cluster
        if len(cluster) == 1:
            keep_indices.append(i)
            continue

        # --- Strategy to pick best representative ---

        subdf = df.iloc[cluster]

        if date_column in df.columns:
            # If dates available, prefer earliest date
            if subdf[date_column].notna().any():
                earliest_date_idx = subdf[date_column].sort_values().index[0]
                keep_indices.append(earliest_date_idx)
                continue

        # fallback: keep the longest text
        longest_idx = subdf[text_column].str.len().idxmax()
        keep_indices.append(longest_idx)

    # return reduced dataframe
    return df.loc[keep_indices].reset_index(drop=True)

def normalize_publisher(name):
    """
    Normalize Italian publishers while preserving local newspaper identities.
    This function is tailored for your dataset.
    Returns clean lowercase publisher names.
    """

    if not isinstance(name, str):
        return name

    n = name.strip().lower()

    # Remove URL artifacts
    n = n.replace("httpswww", "")
    n = n.replace("http", "").replace("https", "")
    n = n.replace(".it", "").replace(".com", "").replace(".net", "")
    n = re.sub(r"\s+", " ", n).strip()

    # -------------------------
    # 1. CORRIERE DELLA SERA (national)
    # -------------------------
    # Match standalone "corriere"
    if n == "corriere":
        return "corriere della sera"

    # Typos in dataset
    if n == "corriere della ser":
        return "corriere della sera"

    # Keep LOCAL Corriere titles untouched:
    # - corriere cesenate
    # - corriere roma
    # - corriere romagna
    # - corriere fiorentino
    # - corriereadriaticoit → corriere adriatico
    if any(local in n for local in [
        "corriere cesenate",
        "corriere roma",
        "corriere romagna",
        "corriere fiorentino",
        "corriere adriatico"
    ]):
        n = n.replace("corriereadriaticoit", "corriere adriatico")
        return n

    # -------------------------
    # 2. LA REPUBBLICA
    # -------------------------
    if "repubblica" in n:
        return "la repubblica"

    # -------------------------
    # 3. IL SOLE 24 ORE (fix typo: "il sole  ore")
    # -------------------------
    if n.startswith("il sole"):
        return "il sole 24 ore"

    # -------------------------
    # 4. IL METEO network
    # -------------------------
    if "meteo" in n:
        # Make them consistent
        if "ilmeteo" in n:
            return "ilmeteo"
        return n

    # -------------------------
    # 5. ANSA (many variants)
    # -------------------------
    if n.startswith("ansa"):
        return "ansa"
    if "agenzia ansa" in n:
        return "ansa"

    # -------------------------
    # 6. TODAY.it LOCAL NETWORK
    # Examples:
    #   ravennatoday, cesenatoday, udinetoday, baritoday…
    # -------------------------
    if n.endswith("today"):
        return n

    # -------------------------
    # 7. IL FATTO QUOTIDIANO (and foundation)
    # -------------------------
    if "fatto quotidiano" in n:
        return "il fatto quotididano"

    # -------------------------
    # 8. LA STAMPA
    # -------------------------
    if n.startswith("la stampa"):
        return "la stampa"

    # -------------------------
    # 9. IL TEMPO
    # -------------------------
    if n == "il tempo":
        return "il tempo"

    # -------------------------
    # 10. Fanpage (fix fanpageit → fanpage)
    # -------------------------
    if n.startswith("fanpage"):
        return "fanpage"

    # -------------------------
    # 11. GENERAL CLEANUP
    # Remove trailing "it"
    # e.g. ilgiornaleit → ilgiornale
    # -------------------------
    if n.endswith("it") and len(n) > 4:
        n = n[:-2]

    return n


# %%
if __name__ == "__main__":
    # Set path
    ROOT = Path(__file__).resolve().parent.parent 
    path = ROOT / "data"

    # read gnews data
    gnews_file = "alluvione emilia-romagna-2023-04-01-to-2024-03-01.csv.gz"
    gnews = pd.read_csv(f"{path}/gnews/{gnews_file}", compression="gzip")

    # %%
    # prepare gnews data
    columns = ["title", "published date", "publisher_name", "text"]
    gnews = gnews[columns]
    gnews = gnews.rename(columns={"published date": "date", "publisher_name": "publisher"})
    gnews["source"] = "gnews"
    gnews["date"] = pd.to_datetime(
    gnews["date"],
    format="%a, %d %b %Y %H:%M:%S GMT",
    errors="coerce"
    )
    # %%
    # read lexis-nexis articles
    ln_news = pd.read_csv(f"{path}/lexis_nexis/ln_articles.csv.gz", compression="gzip")
    # prepare gnews data
    ln_news = ln_news.rename(columns={"Title": "title", "Publisher": "publisher", "Body": "text", "Load Date": "date"})
    ln_news = ln_news[["title", "date", "publisher", "text"]]
    ln_news["date"] = pd.to_datetime(ln_news["date"], errors="coerce")
    ln_news["source"] = "lexis-nexis"

    # %%
    # merge gnews and ln_news
    df = pd.concat([gnews, ln_news], ignore_index=True)

    # %% drop the articles with missing text
    df = df.dropna(subset="text").reset_index(drop=True)
    # Clean the 'publisher' column
    df["publisher"] = df["publisher"].str.replace(r" \(Italy\)", "", regex=True)  # Remove ' (Italy)'
    df["publisher"] = df["publisher"].apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", x))  # Remove special characters
    df["publisher"] = df["publisher"].str.lower().str.strip()  # Convert to lowercase and strip the white space

    #%% drop the articles with less than 30 words
    df = df[df["text"].str.split().str.len() >= 30].reset_index(drop=True)

    # %%
    # translate articles into english
    # Initialize tqdm progress bar
    tqdm.pandas(desc="Translating")
    # Translate the text
    df["translated_text"] = df["text"].progress_apply(translate)

    # %%
    # drop articles with high similarity
    df_cleaned = remove_similar_texts(df, text_column="translated_text")
    
    # %%
    # drop articles that are not published between start date and end date
    start_date = datetime.datetime(2023, 4, 1)
    end_date = datetime.datetime(2024, 6, 1)
    # df_cleaned["date"] = pd.to_datetime(df_cleaned["date"], errors="coerce")
    mask = (df_cleaned["date"] >= start_date) & (df_cleaned["date"] <= end_date)
    df_cleaned = df_cleaned[mask].reset_index(drop=True)

    #%%  Add article index
    df_cleaned["article_index"] = np.arange(len(df_cleaned))

    #%% Normalize publisher names
    df_cleaned["publisher"] = df_cleaned["publisher"].apply(normalize_publisher)
    
    #%% Define the cateogories of the publishers
    publisher_category = {

    # ============================================================
    # NATIONAL NEWSPAPERS & MAJOR PRESS AGENCIES
    # ============================================================
    "la repubblica": "national",
    "corriere della sera": "national",
    "corriere": "national",
    "la stampa": "national",
    "il sole 24 ore": "national",
    "ansa": "national",
    "ansa notiziario generale in italiano": "national",
    "ansa business news": "national",
    "ansa financial news": "national",
    "ansa english media service": "national",
    "il fatto quotidiano": "national",
    "il manifesto": "national",
    "il foglio": "national",
    "il giorno": "national",
    "il tempo": "national",
    "il messaggero": "national",
    "il giornale": "national",
    "il giornale d'italia": "national",
    "italiaoggi": "national",
    "milano finanza": "national",
    "mffashion": "national",
    "mf": "national",
    "la nazione": "national",
    "la gazzetta dello sport": "national",
    "quotidiano nazionale": "national",
    "internazionale": "national",  # magazine
    "avvenire": "national",
    "pagella politica": "digital",  # fact-checking, not national

    # Special case: REGIONAL GROUP but widely circulated
    "il resto del carlino": "regional",


    # ============================================================
    # REGIONAL / LOCAL NEWSPAPERS (highly accurate)
    # ============================================================
    "veronasera": "regional",
    "udinetoday": "regional",
    "bolognatoday": "regional",
    "ravennatoday": "regional",
    "liguria oggi": "regional",
    "ravenna e dintorni": "regional",
    "ravennawebtv": "regional",
    "il messaggero veneto": "regional",
    "baritoday": "regional",
    "padovaoggi": "regional",
    "tvprato": "regional",
    "gazzetta matin": "regional",
    "lecceprima": "regional",
    "forltoday": "regional",
    "chietitoday": "regional",
    "livornotoday": "regional",
    "veneziatoday": "regional",
    "triesteprima": "regional",
    "piacenzasera": "regional",
    "modenatoday": "regional",
    "firenzetoday": "regional",
    "toscanaoggi": "regional",
    "emilia romagna news": "regional",
    "la tribuna di treviso": "regional",
    "lecconotizie": "regional",
    "bergamonews": "regional",
    "cesenatoday": "regional",
    "la voce di rovigo": "regional",
    "gazzetta di parma": "regional",
    "il cittadino": "regional",
    "il cittadino di monza e brianza": "regional",
    "corriere fiorentino": "regional",
    "corriere roma": "regional",
    "corriere cesenate": "regional",
    "corriere romagna": "regional",
    "ilgazzettino": "regional",
    "la nuova provincia asti": "regional",


    # ============================================================
    # DIGITAL / ONLINE-ONLY MEDIA
    # ============================================================
    "meteoweb": "digital",
    "meteoit": "digital",
    "ilmeteo": "digital",
    "bmeteo": "digital",
    "hdblog": "digital",
    "geopop": "digital",
    "fanpage": "digital",
    "fanpageit": "digital",
    "greenme": "digital",
    "greenreport economia ecologica e sviluppo sostenibile": "digital",
    "tassefiscocom": "digital",
    "altreconomia": "digital",
    "il post": "digital",
    "open": "digital",
    "infoaut": "digital",
    "today": "digital",
    "todayit": "digital",
    "freedompress": "digital",
    "idealista": "digital",
    "meteosvizzera": "digital",
    "eunews": "digital",
    "europatoday": "digital",
    "exibart": "digital",
    "ability channel": "digital",
    "lentepubblica": "digital",
    "lavialibera": "digital",
    "start magazine": "digital",
    "dissapore": "digital",
    "fruitbook magazine": "digital",
    "notiziediprato": "digital",
    "ricettasprint": "digital",
    "formulapassion": "digital",
    "facta": "digital",
    "terzobinario": "digital",
    "collettiva": "digital",
    "ti consiglio": "digital",
    "tempi": "digital",
    "ohga": "digital",
    "stylepiccoli": "digital",
    "rolling stone italia": "digital",
    "movieplayer": "digital",
    "dmove": "digital",
    "sicurauto": "digital",
    "vaielettrico": "digital",
    "globalist": "digital",
    "nove da firenze": "regional",  # CORRECTED


    # ============================================================
    # TV / RADIO / MULTIMEDIA
    # ============================================================
    "sky tg": "tv_radio",
    "sky sport": "tv_radio",
    "radio popolare": "tv_radio",
    "radio colonna": "tv_radio",
    "vatican news italiano": "tv_radio",
    "tv sorrisi e canzoni": "tv_radio",
    "teleambiente tv": "tv_radio",
    "radio citt fujiko": "tv_radio",


    # ============================================================
    # SCIENCE / WEATHER / ENVIRONMENT
    # ============================================================
    "tempo italia": "science_weather",
    "centro meteo emilia romagna": "science_weather",
    "greenreport": "science_weather",
    "ambiente sicurezza web": "science_weather",
    "ambienteinforma": "science_weather",


    # ============================================================
    # GOVERNMENT / MUNICIPALITY / PUBLIC INSTITUTIONS
    # ============================================================
    "protezione civile": "institutional",
    "protezione civile emilia romagna": "institutional",
    "comune di livorno": "institutional",
    "comune di cervia": "institutional",
    "comune di castenaso": "institutional",
    "comune di ozzano dellemilia": "institutional",
    "roma capitale": "institutional",
    "regione emiliaromagna": "institutional",
    "regione emiliaromagna salute": "institutional",
    "regione emiliaromagna energia": "institutional",
    "regione emilia romagna": "institutional",
    "rappresentanza in italia della commissione europea": "institutional",
    "european space agency": "institutional",
    "ministero della cultura": "institutional",
    "scuola regione emilia romagna": "institutional",
    "mobilit regione emilia romagna": "institutional",


    # ============================================================
    # NGO / ASSOCIATION / RELIGIOUS
    # ============================================================
    "wwf italia": "ngo_association",
    "save the children italia": "ngo_association",
    "legambiente": "ngo_association",
    "legambiente emiliaromagna": "ngo_association",
    "caritas italiana": "ngo_association",
    "comunione e liberazione": "ngo_association",
    "associazione nazionale alpini": "ngo_association",
    "consulenti del lavoro": "ngo_association",
    "cgil": "ngo_association",
    "federconsumatori": "ngo_association",
    "libertas san marino": "ngo_association",
    "diocesi di milano": "ngo_association",
    "famiglia cristiana": "ngo_association",


    # ============================================================
    # BUSINESS / FINANCE / CORPORATE
    # ============================================================
    "gruppo intesa sanpaolo": "business_finance",
    "confindustria": "business_finance",
    "quifinanza": "business_finance",
    "firstonline": "business_finance",
    "dm distribuzione moderna": "business_finance",
    "simplybiz dedicato a chi opera nel mondo del credito": "business_finance",
    "fashionmagazine": "business_finance",
    "bmw group pressclub": "business_finance",
    "ipsoa": "business_finance",
    "fiscooggi": "business_finance",


    # ============================================================
    # OTHER / UNCATEGORIZED

    "metro": "other",
    "elle": "other",
    "vita": "other",
    "food": "other",
    "motor valley": "other",
    "ufficio stampa": "other",
}


    df_cleaned["publisher_category"] = df_cleaned["publisher"].map(publisher_category).fillna("other")

    # %%
    # Export prepared data
    df_cleaned.to_csv(f"{path}/news_corpus/news_corpus.csv.gz", compression="gzip", index=False)
    # %%
