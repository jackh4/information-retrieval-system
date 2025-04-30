# Job Search Query System with NLP and Semantic Search

## Overview

Allows users to search for jobs by entering natural language queries.
Uses NLP techniques and models to understand the user's intent, extract key phrases,
expand queries with synonyms, and retrieve the most relevant job postings using semantic similarity.

## Main Features

1. **Job Posting Ingestion**:

   - Loads and cleans a dataset of job postings (CSV format).
   - Combines title, description, and skill descriptions for semantic indexing.

2. **Keyphrase Extraction**:

   - Uses RAKE (Rapid Automatic Keyword Extraction) to extract meaningful phrases from user queries.

3. **NLI-Based Sentiment Classification**:

   - Uses a pre-trained Natural Language Inference (NLI) model to classify key terms into inclusive and exclusive categories, based on the relationship between the query and job terms.

4. **Query Expansion via Word2Vec**:

   - Expands inclusive search terms by finding semantically similar words using Word2Vec and FAISS for efficient similarity search.

5. **Semantic Ranking**:

   - Jobs are encoded using `SentenceTransformer` and ranked by cosine similarity with the user-expanded query using FAISS.

6. **Interactive Search**:
   - Runs in a CLI loop where the user can type a query and receive the top 25 most similar job listings.

## Prerequisites

- The dataset used can be downloaded at https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
- A dataset file should be located at `./data/postings.csv` (should include `title`, `description`, and `skills_desc` columns).
- Python 3.11+

## Execution

To run the program:

```bash
pip install -r requirements.txt
python retrieval.py
```
