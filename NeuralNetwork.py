import numpy as np
import nltk #for NLP
from nltk.stem.porter import PorterStemmer  #To find radical of words
#nltk.download('punkt')

#We divide text into several words with word_tokenizer() and into several line with sent_tokenizer()
Stemmer = PorterStemmer()

jarvis = "hello", "hello"
# 'h', 'e', 'l', 'l', 'o'

# 'h', 'e', 'l', 'l', 'o'

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return  Stemmer.stem(word.lower())  #stem() is used to find the radical of words  

def bag_of_words(tokenized_sentence, words):
    sentence_word = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)

    for idx, w in enumerate(words):

        if w in sentence_word:
            bag[idx] = 1
    return bag

#Return a vector on 0 or 1. if a letter of words is in the radical of tokenized_word, we have 1 if not we have 0
print(bag_of_words(['hello'], 'greeting'))