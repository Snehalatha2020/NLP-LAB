import spacy
nlp = spacy.load("en_core_web_sm")
example_string = "My name is Snehal. My father name is Gunwant. My mother name is Pushplata. I am in final year of Engineering."
doc = nlp(example_string)
print("Sentence Tokenization:")
for sent in doc.sents:
    print(sent.text)
print("\nWord Tokenization:")
for token in doc:
    print(token.text)
print("\nFiltered Words (without stopwords and punctuation):")
filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
print(filtered_tokens)
print("\nLemmatization:")
lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
print(lemmatized_tokens)