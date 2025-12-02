# narrative-roles-attribution

This repository contains materials related to the paper  
**"Narrating the Deluge: Media Framing and Public Attribution of Climate
Change in the Emilia-Romagna Floods"**  
by Tina Comes, Giulia Piccillo, Anmol Soni, and Tania Treibich.

## Overview
The study investigates how the May 2023 floods in Emilia-Romagna, Italy, influenced climate change discourse.  
It compares **media framing** and **public attribution** using narrative character roles (heroes, villains, victims) to highlight asymmetries between newspaper coverage and survey responses. Working draft is available upon request.

## Methods
- Newspaper corpus analysis
- Narrative character-roles framework
- Survey data from flood-affected communes

## Contents
```
pynpf
├─ data
|  ├─ actor_directory             // manually annotated and k-means clustered actor groups
|  ├─ gnews                       // newsarticles downloaded from google-news
|  ├─ lexis_nexis                 // newsarticles downloaded from lexis-nexis
|  ├─ model_performance           // model role classification matrix
|  ├─ news_corpus                 // news corpus used for analysis
|  ├─ predictions                 // results
|  ├─ survey_data                 // survey results from qualtrics
|  ├─ training_data               // annotated data used for training the model
├─ figures                        // main result figures
|  ├─ descriptive                 // newspaper corpus descriptive
├─ models                         // offline models
|  ├─ classifier                  // trained classifier RoBERTa + XG-Boost model
|  ├─ roberta-sentiment           // RoBERTa sentiment analysis model
├─ ReadMe.md
├─ requirements.txt
└─ scripts                        // The scripts should be executed in following sequence
   ├─ lexis_nexis_to_dataframe.py // create dataframe from articles downloaded from Lexis-Nexis
   ├─ gnews_article_scraper.py    // download the articles from Google News
   ├─ news_data_preparation.py    // clean and prepare the text corpus
   ├─ named_entity_recognition.py // extract named enities from the corpus for actor clustering
   ├─ extract_svos.py             // split the corpus into sentences and extract Subject-Verb-Objects
   ├─ sample_for_annotation.py    // sample sentences for annotation
   ├─ model_training.py           // train model to extract narrative character roles
   ├─ predict_corpus.py           // extract narrative character roles for the entire corpus
   ├─ results.py                  // analyse and visualise the results

  Note that `data` and `models` folders is not uploaded in the git repo.
```

## Key Findings
- Media emphasized immediate impacts and institutional responses.
- Residents, particularly from floods affected commmunes more frequently attributed floods to climate change.
- Attribution to climate change correlated with pro-environmental behavioral shifts.
