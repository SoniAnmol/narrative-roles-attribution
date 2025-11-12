"""Uses relatio library to mine entities from text"""

# %% Import libraries
import pandas as pd
from relatio import Preprocessor
from extract_svos import set_preprocessor, split_sentences

# %% main
if __name__ == "__main__":
    # Read data
    df = pd.read_csv("../data/alluvione_emilia-romagna_2023-04-01-to-2024-03-01_summarized_translated_cleaned.csv")

    # %% Set preprocessor
    p = set_preprocessor()

    # %% Split into sentences
    df = split_sentences(df, p)

    # %% Mine known entities
    known_entities = p.mine_entities(df["sentence"], clean_entities=True, progress_bar=True)

    # %% Export known entities
    # Convert the Counter to a DataFrame
    entities = pd.DataFrame.from_dict(known_entities, orient="index", columns=["Count"])

    # %% Reset the index to make the Counter keys a column
    entities = entities.reset_index().rename(columns={"index": "Item"})
    
    # %% Export the DataFrame to an Excel file
    entities.to_excel("../output/entities_news.xlsx", index=False)
