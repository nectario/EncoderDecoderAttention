
from gensim.corpora.dictionary import Dictionary
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from tqdm import tqdm
import spacy
from spacy.lang.en import English
from utils.data_utils import convert_to_string

class GensimGloveVectorizer():

    reserved = ["<PAD>", "<UNK>"]
    embedding_matrix = None
    embedding_word_vector = {}
    glove = False

    def __init__(self, reserved=None, min_occur=1, glove_path=None):

        self.mi_occur = min_occur

        self.nlp = spacy.load("en")
        self.tokenizer = English().Defaults.create_tokenizer(self.nlp)

        if glove_path is not None:
            self.load_glove(glove_path)
            self.glove = True

        if reserved is not None:
            self.vocabulary = Dictionary([self.reserved.extend(reserved)])
        else:
            self.vocabulary = Dictionary([self.reserved])

    def get_vocabulary_size(self):
        return len(self.vocabulary.token2id.items())

    def load_glove(self, glove_file_path):
        f = open(glove_file_path, encoding="utf-8")
        for line in tqdm(f):
            value = line.split(" ")
            word = value[0]
            coef = np.array(value[1:], dtype='float32')
            self.embedding_word_vector[word] = coef
        f.close()

    def get_embedding_matrix(self):
        return self.embedding_matrix

    def is_word(self, string_value):
        if self.embedding_word_vector.get(string_value):
            return True

    def fit(self, X):
        x_tokens = []
        for x in X:
            if x is not None:
                #x_tokens.append(word_tokenize(x.lower()))
                tokens = self.tokenizer(x.lower())
                word_str_tokens =  list(map(convert_to_string,tokens))
                x_tokens.append(word_str_tokens)
                self.vocabulary.add_documents(x_tokens)

        self.vocabulary.filter_extremes(no_below=self.mi_occur, no_above=1.0, keep_tokens=self.reserved)

        if self.glove:
            print("Vocabulary Size:",self.get_vocabulary_size())
            self.embedding_matrix = np.zeros((self.get_vocabulary_size(), 300))
            for word, i in tqdm(self.vocabulary.token2id.items()):
                if word == "<PAD>":
                    embedding_value = np.zeros((1, 300))
                elif word == "<UNK>":
                    sd =  1/np.sqrt(300)
                    np.random.seed(seed=42)
                    embedding_value = np.random.normal(0, scale=sd, size=[1, 300])
                else:
                    embedding_value = self.embedding_word_vector.get(word)
                    if embedding_value is None:
                        embedding_value = self.embedding_word_vector.get("<UNK>")

                if embedding_value is not None:
                    self.embedding_matrix[i] = embedding_value

        return  self.transform(x_tokens)

    def transform(self, X):
        #X = X.tolist()
        if hasattr(self, "limit"):
            return [[i if i < self.limit else self.reserved.index("<UNK>")
                for i in self.vocabulary.doc2idx(x, unknown_word_index=self.reserved.index("<UNK>"))]
                for x in X]
        else:
            return [self.vocabulary.doc2idx(x, unknown_word_index=self.reserved.index("<UNK>")) for x in X]


    def inverse_transform(self, X):
        return [[ self.vocabulary[i] for i in x ] for x in X]

    def save(self, file_path="./vecorizer.vec"):
        with open(file_path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return file_path

    @classmethod
    def load(cls, file_path):
        with open(file_path, "rb") as handle:
            self = pickle.load(handle)
        return self