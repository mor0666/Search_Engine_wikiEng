import math
from contextlib import closing

import nltk
# from google.cloud import storage
import re
import pickle
from nltk.corpus import stopwords
from pathlib import Path

from inverted_index_gcp import MultiFileReader

#tokenize function
nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)


def tokenize(text):
    return [token.group() for token in RE_WORD.finditer(text.lower())]


TUPLE_SIZE = 6
#reading the post list from the index of specific word
def read_posting_list(inverted, w):
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list

#calculate cosine similarity using tf-idf
def cossim(query, index, dict_Dl):
    result_dict = {}
    query_vector = {}
    len_q= len(query)
    for idDoc in dict_Dl.keys():
        result_dict[idDoc]=float(0)
    for word in query: #make query as vector
        if(word not in query_vector):
            query_vector[word] =0
        query_vector[word] += 1 / len_q
    temp_vec_q= query_vector.values()
    norm=0.0
    for i in temp_vec_q: #normalize the query
        norm= norm+i**2
    norm_r = math.sqrt(norm)
    word_tf_dict={}
    inner=0.0
    for term in query: #calculate cossim
        try:
            idf = math.log2(len(dict_Dl) / index.df[term])
        except:
            idf=0
        tf_word = query_vector[term]
        try:
            pls = read_posting_list(index, term)
            pls=pls
            # print(pls)
        except:
            pls=[]
        for id, tf in pls:
            result_dict[id]+= idf*tf
            try:
                word_tf_dict[id][term]=tf/dict_Dl[id]
            except:
                g=[]
        for k, v in word_tf_dict.items():
            for key, val in v.items():
                inner+=query_vector[term]*word_tf_dict.get(key)
                inner = idf*inner
            result_dict[id] = (inner / (dict_Dl[id] * norm_r)) * (len(word_tf_dict))
    return result_dict

  #read index
def read_index(base_dir, name):
    with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
        return pickle.load(f)


def get_posting_gen(index, word):
    words, pls = zip(index.read_posting_list(word))
    return (words, pls)
