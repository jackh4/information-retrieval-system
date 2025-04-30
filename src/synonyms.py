import faiss
import numpy as np
from gensim.models import Word2Vec
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from nltk.corpus import stopwords

def train_word2vec(df):
    # normalize column names for dataframe for further implementation
    tokenized_sentences = df['combined_text'].apply(lambda x: x.lower().split()).tolist()

    stop_words = set(stopwords.words('english'))
    tokenized_sentences = [[w for w in sent if w not in stop_words] for sent in tokenized_sentences]

    phrases = Phrases(tokenized_sentences, min_count=1, threshold=100)
    phraser = Phraser(phrases)
    sentences_with_phrases = [phraser[sent] for sent in tokenized_sentences]

    model = Word2Vec(
        sentences=sentences_with_phrases, 
        vector_size=150, 
        window=5, 
        min_count=3, 
        workers=4, 
    )
    return model, phraser

def build_faiss_index(model):
    words = list(model.wv.index_to_key)
    vectors = np.array([model.wv[word] for word in words]).astype('float32')
    faiss.normalize_L2(vectors) 

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, words

def search_similar(word, model, index, words, phraser, top_k=3):
    phrase_word = phraser[[w.lower() for w in word.split()]][0]

    if phrase_word not in model.wv:
        print(f"'{word}' not in vocabulary as '{phrase_word}'.")
        return []

    vector = np.array([model.wv[phrase_word]]).astype('float32')
    distances, indices = index.search(vector, top_k)

    return [(words[i], distances[0][j]) for j, i in enumerate(indices[0])]