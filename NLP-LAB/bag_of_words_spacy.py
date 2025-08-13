import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

paragraph = """The news mentioned here is fake. Audience do not encourage fake news. Fake news is false or misleading"""

# Process text
doc = nlp(paragraph)

# Preprocess each sentence
corpus = [
    " ".join([token.lemma_ for token in sent if not token.is_stop and token.is_alpha])
    for sent in doc.sents
]

print("Cleaned Corpus:", corpus)

# Bag of Words
cv = CountVectorizer()
print("\nCount Vectorizer Features:\n", cv.fit_transform(corpus).toarray())

# TF-IDF
tfidf = TfidfVectorizer()
print("\nTF-IDF Features:\n", tfidf.fit_transform(corpus).toarray())
