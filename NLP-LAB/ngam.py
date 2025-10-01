from nltk.util import ngrams

n = 1
sentence = "Artificial intelligence is transforming the way humans interact with technology and shaping the future of industries."
unigrams = ngrams(sentence.split(), n)
print(f"\n***********   UNIGRAM    ************************")
for item in unigrams:
    print(item)

n = 2
sentence = "Artificial intelligence is transforming the way humans interact with technology and shaping the future of industries."
unigrams = ngrams(sentence.split(), n)
print(f"\n***********   BIGRAM    ************************")
for item in unigrams:
    print(item)

n = 3
sentence = "Artificial intelligence is transforming the way humans interact with technology and shaping the future of industries."
unigrams = ngrams(sentence.split(), n)
print(f"\n***********   TRIGRAM    ************************")
for item in unigrams:
    print(item)
