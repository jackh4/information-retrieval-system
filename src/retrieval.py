import os
import re
import pandas as pd
import torch
import nltk
import faiss
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from keyterms import extract_key_phrases
from nli_checker import load_nli_model, predict_relation
from synonyms import build_faiss_index, train_word2vec, search_similar

FILE_PATH = './data/postings.csv'
NLP_MODEL = 'all-MiniLM-L6-v2'

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^a-zA-Z\s_]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def load_data(file_path):
    # Can be changed appropriately according to usage
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')

    df = pd.read_csv(file_path)
    df = df.dropna(subset=['description'])
    df['combined_text'] = df['title'].astype(str) + ' ' + df['description'].astype(str) + ' ' + df['skills_desc'].astype(str)
    df['combined_text'] = df['combined_text'].apply(clean_text)
    # for testing
    df = df.sample(n=5000, random_state=42)
    return df

def classify_sentiment(nli_model, query, keyterms):
    inclusive_terms = []
    exclusive_terms = []

    for term in keyterms:
        prediction = predict_relation(nli_model, query, f"jobs with {term}")
        if prediction == 'neutral' or prediction == 'entailment':
            inclusive_terms.append(term)
        else:
            exclusive_terms.append(term)
    return inclusive_terms, exclusive_terms 

def expand_query(terms, model, index, words, phraser):
    expanded_inclusive_terms = []
    for term in terms:
        synonym_terms = search_similar(term, model, index, words, phraser)
        expanded_inclusive_terms.extend([sim_term for sim_term, _ in synonym_terms])
    expanded_inclusive_terms = list(set(expanded_inclusive_terms))
    return expanded_inclusive_terms

def rank_jobs(model, original_terms, synonym_terms, df, job_embeddings):
    # boost weights of original keyterms over their synonym keyterms
    # helps retain original intent while enabling semantic matching
    composite_query = (" ".join(original_terms) + " ") * 5 + (" ".join(synonym_terms) + " ") * 1
    query_embedding = model.encode([composite_query])
    faiss.normalize_L2(query_embedding)
    
    index = faiss.IndexFlatIP(job_embeddings.shape[1])
    index.add(job_embeddings)
    D, I = index.search(query_embedding, 25)

    top_jobs = df.iloc[I[0]].copy()
    top_jobs['similarity'] = D[0]
    return top_jobs

def print_results(ranked_jobs):
    if ranked_jobs.empty:
        print("\nNo jobs found.")
        return

    print("\nTop Matching Jobs:\n")
    for idx, row in ranked_jobs.iterrows():
        print(f"Title: {row['title']}")
        print(f"Similarity Score: {row['similarity']:.4f}")
        print(f"Description: {row['description'][:150]}...")
        print('-' * 50 + '\n\n')

def retrieval():
    df = load_data(FILE_PATH)
    print(f"Loaded {len(df)} job postings")

    word2vec_model, phraser = train_word2vec(df)
    index, words = build_faiss_index(word2vec_model)
    nli_model = load_nli_model()

    nlp_model = SentenceTransformer(NLP_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nlp_model = nlp_model.to(device)

    print("Encoding all job postings...")
    job_embeddings = nlp_model.encode(df['combined_text'].tolist(), show_progress_bar=True, batch_size=64)
    faiss.normalize_L2(job_embeddings)

    job_index = faiss.IndexFlatIP(job_embeddings.shape[1])
    job_index.add(job_embeddings)
    print("FAISS index built for job postings.")

    while True:
        raw_query = input("Search occupations (or 'exit' to quit): ").strip()
        if raw_query.lower() == 'exit':
            break

        keyterms = extract_key_phrases(raw_query)
        if not keyterms:
            print("Search error: keyterms not found")
            continue

        # filtering exclusion terms currently has not been implemented
        inclusive_terms, exclusive_terms = classify_sentiment(nli_model, raw_query, keyterms)

        expanded_inclusive_terms = expand_query(inclusive_terms, word2vec_model, index, words, phraser)

        ranked_jobs = rank_jobs(nlp_model, inclusive_terms, expanded_inclusive_terms, df, job_embeddings)
        print_results(ranked_jobs)

if __name__ == '__main__':
    retrieval()