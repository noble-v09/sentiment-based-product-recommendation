import string
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()

def preprocess_text(X):
    X = X.str.lower()
    X = X.str.replace(f'[{string.punctuation}]', '', regex=True)
    X = X.str.replace('\w*\d\w*', '', regex=True)
    X = X.str.replace('\s\s+', '', regex=True)
    return X

def get_wordnet_tag(tag):
    if tag.startswith('J'):
        return wordnet.ADJ ## Adjectives
    elif tag.startswith('N'):
        return wordnet.NOUN ## Nouns
    elif tag.startswith('R'):
        return wordnet.ADV ## Adverbs
    elif tag.startswith('V'):
        return wordnet.VERB ## Verbs
    else:
        return None

def lemmatize_text(x):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(x))
    wordnet_tagged = map(lambda x: (x[0], get_wordnet_tag(x[1])), nltk_tagged)
    lemma_text_list = list()
    for word, tag in wordnet_tagged:
        if tag:
            lemma_text_list.append(lemmatizer.lemmatize(word, tag))
        else:
            lemma_text_list.append(word)
    return ' '.join(lemma_text_list)