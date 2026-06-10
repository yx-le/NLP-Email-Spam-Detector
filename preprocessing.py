import html
import re

import nltk
from bs4 import BeautifulSoup
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def ensure_nltk_resources():
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
        "taggers/averaged_perceptron_tagger_eng":
            "averaged_perceptron_tagger_eng",
    }

    for resource_path, download_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_name, quiet=True)


ensure_nltk_resources()

LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english")) - {"no", "nor", "not"}


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess_text(text):
    text = html.unescape(str(text))
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [
        token
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]

    tagged_tokens = pos_tag(tokens)
    lemmas = [
        LEMMATIZER.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged_tokens
    ]
    return " ".join(lemmas)
