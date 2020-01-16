from nltk.corpus import words
import pandas as pd
from spellchecker import SpellChecker
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
import os

from symspellpy.symspellpy import SymSpell
from spacy.lang.en import English
import spacy

def get_sym_spell():
    # maximum edit distance per dictionary precalculation
    max_edit_distance_dictionary = 0
    prefix_length = 7
    # create object
    sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
    # load dictionary
    dictionary_path = os.path.join(os.path.dirname(__file__), "../data/dictionaries/frequency_dictionary_en_82_765.txt")
    term_index = 0  # column of the term in the dictionary text file
    count_index = 1  # column of the term frequency in the dictionary text file
    if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
        print("Dictionary file not found")
        return

    return sym_spell

    # a sentence without any spaces
    #input_term = "thequickbrownfoxjumpsoverthelazydog"

    #result = sym_spell.word_segmentation(input_term)
    # display suggestion term, term frequency, and edit distance
    #print("{}, {}, {}".format(result.corrected_string, result.distance_sum,
    #                          result.log_prob_sum))

def replace_chars(text, string=""):
    return text.replace(".", string).replace(",", string).replace("'", string).replace('"', string).replace("(", string).replace(")", string).replace(":", string).\
        replace(";", string).replace("[", string).replace("]", string)

def convert_to_string(token):
    return str(token.text)

def check_spelling(words_in_text, spellchecker):
    word_list = []
    misspelled_words = spellchecker.unknown(words_in_text)
    for word in words_in_text:
        if word.lower() in misspelled_words:
            word_list.append(word)
    return word_list

def is_all_lowercase(value):
    lowercase = [c for c in value if c.islower()]
    if len(value) == len(lowercase):
        return True
    else:
        return False

def check_for_bad_words(file_path):
    spellchecker = SpellChecker()
    input_df = pd.read_excel(file_path)
    ignore = ["LIBOR", "LIBOR01", "LIBO", "BBAs","1-A-1","Moneyline","'s","1", "2", "3", "4","5","6","7","8","9","0","", "telerate", "Telerate","pass-through", "libo","2-A-1","1-A-4","1-A-3","2-A-2","1/32%","1a2","1a1","11:00","annum","servicer","1/16",]
    bad_words = {}
    sym_spell = get_sym_spell()
    corrected_words = {}

    nlp = spacy.load("en")
    tokenizer = English().Defaults.create_tokenizer(nlp)

    for i, row in input_df.iterrows():
        text = str(row["Text"])

        tokens_in_text = tokenizer(text) #word_tokenize(text) #tokenizer.tokenize(text) #text.split(" ") #list(map(replace_chars, text.split(" ")))  #word_tokenize(text)
        words_in_text = list(map(convert_to_string,tokens_in_text))

        misspelled = check_spelling(words_in_text,spellchecker)

        for word in misspelled:
            #word #word.replace(".","").replace(",","").replace("-","").replace("'","").replace('"','').replace("(","").replace(")","").replace(":","").replace(";","").replace("[","").replace("]","")
            if word not in ignore and not word.isdigit() and is_all_lowercase(word): #and dict.check(new_word):
                result = sym_spell.word_segmentation(word)
                corrected_word = result.corrected_string
                print(word)
                count = bad_words.get(word)
                if count is None:
                    bad_words[word] = 1
                    corrected_words[word] = corrected_word
                else:
                    bad_words[word] = count + 1

        print(i, "of", input_df.shape[0])

    return bad_words, corrected_words

if __name__ == "__main__":
    bad_words, corrected_words = check_for_bad_words("../data/RMBS.xlsx")
    #corrected_texts = check_for_bad_words("../data/RMBS.xlsx")

    output_bad_words_df = pd.DataFrame()

    output_bad_words_df["Word"] = bad_words.keys()
    #output_bad_words_df["Corrected Words"] = corrected_words
    output_bad_words_df["Count"] = bad_words.values()

    output_bad_words_df.sort_values(by="Count")

    output_bad_words_df.to_excel("../data/RMBS_bad_words.xlsx", index=None)
