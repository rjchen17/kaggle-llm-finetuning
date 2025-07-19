import unicodedata
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer

"""
- preprocesses text by normalizing unicode, converting to lowercase, removing stopwords, and lemmatizing
- returns list of tokens (to be used by word2vec or something)

"""

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def preprocess(text):
    text = unicodedata.normalize('NFKC', text).lower() # gets rid of unicode - unsure if we want this; I picked NFKC but we can change this
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    
    cleaned = []
    for word, pos in tagged_tokens:
        if word in stop_words:                      # maybe this gets rid of too much ("yes", "no", etc)
            continue
        wordnet_pos = get_wordnet_pos(pos)
        lemma = lemmatizer.lemmatize(word, wordnet_pos)
        cleaned.append(lemma)
    return cleaned