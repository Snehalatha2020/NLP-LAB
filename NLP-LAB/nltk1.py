import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

example_string = "My name is Snehal. My father's name is Gunwant. My mother's name is Pushplata. I am currently in final year of my BTech.The syllabus is beautifully designed."

sentences = sent_tokenize(example_string)
print("Sentence Tokenization:")
print(sentences)

words = word_tokenize(example_string)
print("\nWord Tokenization:")
print(words)

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]
print("\nFiltered Words (without stopwords):")
print(filtered_words)

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
print("\nLemmatized Words:")
print(lemmatized_words)
