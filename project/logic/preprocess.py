# Data preprocess, translate slangs, preprocess columns, train test split, embed and pad.
import numpy as np
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import gensim.downloader
from keras.preprocessing.sequence import pad_sequences

from data import ImportData

# Import data from data.py
data_goemotions = ImportData.goemotions()
data_abbreviations = ImportData.abbreviations()
data_slangs = ImportData.slangs()

# Assert correct data import
assert data_goemotions.shape == (207814, 29)
assert data_abbreviations.shape == (114, 2)
assert data_slangs.shape == (3357, 2)

print(f'Data imported correctly ✅')

# Starting preprocess
print (f'Starting preprocess ⏳')


# Concatenating slang and abbreviation datasets
data_slangs.rename(columns = {'acronym':'Abbreviations','expansion':'Text'}, inplace = True)
data_slang_all = pd.concat([data_abbreviations , data_slangs], axis=0)

# Drop duplicates and null values
data_slang_all.drop_duplicates(inplace=True)
data_slang_all.dropna

# Transforming DF into dict for mapping
slang_dict = dict(zip(data_slang_all.Abbreviations, data_slang_all.Text))

# Creating class to make the Slang translation
class SlangTranslation:

  def __init__(self, col):

    self.col = col

  def remove_punctuation(self, txt):
    """Iterates through each word of the string and removes punctuation"""
    txt = txt.lower()

    for punctuation in string.punctuation:
        txt = txt.replace(punctuation, ' ')

    return txt

  def string_translator(self, txt):
    """Iterates through each word of the string and translates them"""

    txt = ' '.join([slang_dict.get(i, i) for i in txt.split()])

    return txt

  def apply_translator(self):
    """Takes the text column as input, outputs the same column translated."""

    txt = self.col.apply(self.remove_punctuation)

    txt = txt.apply(self.string_translator)

    return txt

# Creating class for basic preprocess
class PreprocessingText:

    def __init__(self, col):
        self.col = col

    def cleaning_text(self, txt):
        """
        Transform everything in lowercase, strip the text and remove everything that's not letters or space
        """
        txt = txt.lower()
        txt = txt.replace('’', '').strip()
        text_cleaned = ''.join(filter(lambda x: x.isalpha() or x.isspace(), txt)) # Remove everything that's not letters or space

        return text_cleaned

    def tokenizing_text(self, txt):
        """
        Create stopword list, tokenize the words and return the text tokenized without stopwords
        """
        stop_words = set(stopwords.words('english')) # Create stopword list
        tokenized = word_tokenize(txt) # Tokenize
        tokenized_text = [word for word in tokenized if not word in stop_words] # Tokenizing text

        return tokenized_text

    def lemmatizing_text(self, tokenized_txt):
        """
        Lemmatize the text and return a cleaned sentence
        """
        lemmatized = [
            WordNetLemmatizer().lemmatize(word, pos='v') for word in tokenized_txt
        ] # Lematize
        cleaned_sentence = " ".join(word for word in lemmatized) # Lemmatized text

        return cleaned_sentence

    def apply_preprocessor(self):
        """
        Apply all functions above and drop 'na' values
        """
        txt = self.col
        txt = txt.apply(self.cleaning_text)
        txt = txt.apply(self.tokenizing_text)
        txt = txt.apply(self.lemmatizing_text)
        txt = txt.dropna()

        return txt

# Applying the slang translation

data_goemotions['text'] = SlangTranslation(data_goemotions['text']).apply_translator()
data_preprocessed = data_goemotions
data_preprocessed['text'] = PreprocessingText(data_preprocessed['text']).apply_preprocessor()

# Assert Dataset is preprocessed
assert data_preprocessed.shape == (207814, 29)
assert data_preprocessed['text'][16] == 'well id say pretty good chance girl laugh loud'

print(f'Data preprocessed correctly ✅')

print(f'Starting train test split ⏳')

# Splitting train test
X = data_preprocessed['text']
y = data_preprocessed.drop(columns= 'text')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

# Assert correct split shapes
assert X_train.shape == (166251,)
assert X_test.shape == (41563,)
assert y_train.shape == (166251, 28)
assert y_test.shape == (41563, 28)

print(f'Data splitted correctly ✅')

print(f'Starting embedding ⏳')

# Embedding
# word2vec = gensim.downloader.load('glove-wiki-gigaword-50')

# class EmbeddingText:
#     def __init__(self, word2vec):
#         """
#         Initialize the EmbeddingText with a word2vec model.
#         """
#         self.word2vec = word2vec

#     def embed_sentence(self, sentence):
#         """
#         Convert a sentence (list of words) into a matrix representing the words in the embedding space.
#         """
#         embedded_sentence = [self.word2vec[word] for word in sentence if word in self.word2vec]
#         return np.array(embedded_sentence)

#     def embed_sentences(self, sentences):
#         """
#         Convert a list of sentences into a list of matrices.
#         """
#         return [self.embed_sentence(sentence) for sentence in sentences]

# embedding_instance = EmbeddingText(word2vec)
# X_train_embedded = embedding_instance.embed_sentences(X_train)

# assert len(X_train_embedded) == 166251

# print(f'Data embedded ✅')

# print(f'Starting padding ⏳')

# X_pad = pad_sequences(X_train_embedded, dtype='float16', padding='pre')
