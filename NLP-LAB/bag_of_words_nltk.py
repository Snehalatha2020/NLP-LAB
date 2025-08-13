import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Download required datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

paragraph = """The news mentioned here is fake. Audience do not encourage fake news. Fake news is false or misleading"""

# Tokenizing into sentences
sentences = nltk.sent_tokenize(paragraph)

lemmatizer = WordNetLemmatizer()
corpus = []

for i in range(len(sentences)):
    sent = re.sub('[^a-zA-Z]', ' ', sentences[i])  # remove non-alphabets
    sent = sent.lower()  # lowercase
    sent = sent.split()  # split into words
    sent = [lemmatizer.lemmatize(word) for word in sent if word not in set(stopwords.words('english'))]
    sent = ' '.join(sent)
    corpus.append(sent)

print("Cleaned Corpus:", corpus)

# Bag of Words
cv = CountVectorizer()
independentFeatures = cv.fit_transform(corpus).toarray()
print("\nCount Vectorizer Features:\n", independentFeatures)

# TF-IDF
tfidf = TfidfVectorizer()
independentFeatures_tfIDF = tfidf.fit_transform(corpus).toarray()

# Convert TF-IDF to DataFrame for table visualization
tfidf_df = pd.DataFrame(independentFeatures_tfIDF, columns=tfidf.get_feature_names_out())

print("\nTF-IDF Table:\n")
print(tfidf_df)
