from rake_nltk import Rake
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# custom word list not included within nltk stop_words, can be expanded requiring data on user queries
remove_words = {"involve", "need", "require", "knows", "someone"}

def clean_phrases(phrases):
    stop_words = set(stopwords.words('english'))
    cleaned = []
    for phrase in phrases:
        words = word_tokenize(phrase)
        filtered_words = [word for word in words if word.lower() not in stop_words and word.lower() not in remove_words]
        if filtered_words:
            cleaned.append(' '.join(filtered_words))
    return cleaned

def extract_key_phrases(query):
    rake = Rake()
    rake.extract_keywords_from_text(query)
    ranked_phrases = rake.get_ranked_phrases()
    return clean_phrases(ranked_phrases)