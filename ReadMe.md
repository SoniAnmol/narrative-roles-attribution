# narrative-roles-attribution

This repository contains materials related to the paper  
**"Narrating the Deluge: Media Framing and Public Attribution of Climate
Change in the Emilia-Romagna Floods"**  
by Tina Comes, Giulia Piccillo, Anmol Soni, and Tania Treibich.

## Overview
The study investigates how the May 2023 floods in Emilia-Romagna, Italy, influenced climate change discourse.  
It compares **media framing** and **public attribution** using narrative character roles (heroes, villains, victims) to highlight asymmetries between newspaper coverage and survey responses. Working draft is available upon request.

## Contents
```
pynpf
├─ figures
├─ ReadMe.md
├─ requirements.txt
└─ scripts
   ├─ lexis_nexis_to_dataframe.py
   ├─ gnews_article_scraper.py
   ├─ news_data_preparation.py
   ├─ extract_svos.py
   ├─ sample_for_annotation.py
   ├─ training_model.py
   └─ visualisations

```

## Methods
- Newspaper corpus analysis
- Narrative character-roles framework
- Survey data from flood-affected communes

## Sequence of running scripts
The scripts must be executed in the following order:
1. **gnews_articles_scraper.py** Download the articles from Google News
1. **lexis_nexis_to_dataframe.py** Create dataframe from articles downloaded from Lexis-Nexis 
1. **news_data_preparation.py** Clean and prepare the text corpus
<!-- 1. **named_entitiy_recognition.py** Extract named enities from the corpus -->
1. **extract_svos.py** Split the corpus into sentences and extract Subject-Verb-Objects
1. **sample_data_for_annotation.py** Sample sentences for annotation
1. **training_model.py** Train model to classify Narrative Character Roles

## Key Findings
- Media emphasized immediate impacts and institutional responses.
- Residents, particularly from floods affected commmunes more frequently attributed floods to climate change.
- Attribution to climate change correlated with pro-environmental behavioral shifts.


