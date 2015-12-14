import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

def build_data_cv(dataFiles, cv=10, minVocab=5):
    """
    Loads data and splits into cv folds.
    """
    revs = []
    vocab = defaultdict(float)
    y_ = 0
    for textfile in dataFiles:
        with open(textfile, "rb") as f:
            for line in f:
                if line != "":
                    line = re.sub(r"Mr\.", "Mr", line)
                    line = re.sub(r"Ms\.", "Ms", line)
                    line = re.sub(r"Mrs\.", "Mrs", line)
                    line = re.sub(r" ([A-Z])\. ", r" \1 ", line)
                    line = re.sub(r" ([A-Z])\.([A-Z])\. ", r" \1\2 ", line)
                    line = re.sub(r" ([A-Z])\.([A-Z])\.([A-Z])\. ", r" \1\2\3 ", line)
                    sentences = re.compile("\.[^0-9]").split(line)
                    for sentence in sentences:   
                        clean_sentence = clean_str(sentence)                                     
                        words = set(clean_sentence.split())
                        for word in words:
                            vocab[word] += 1
                        datum  = {"y":y_, 
                        "text": clean_sentence,   
                        "original": sentence,                          
                        "num_words": len(clean_sentence.split()),
                        "split": np.random.randint(0,cv)}
                        revs.append(datum)

        y_ = y_ + 1

    for k in vocab.keys():
        if vocab[k] < minVocab:
            del vocab[k]

    return revs, vocab, y_
    
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  
    word_vecs["UUUKKK"] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


if __name__=="__main__":    
    w2v_file = sys.argv[1]     
    data_folder = ["trump06.csv","trump07.csv","trump08.csv", "trump09.csv", "trump10.csv", "trump11.csv", "trump12.csv"]    
    print "loading data...",        
    revs, vocab, y_ = build_data_cv(data_folder, cv=10, minVocab=5)
    numclasses = y_
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab, k=25)
    W2, _ = get_W(rand_vecs, k=25)
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    cPickle.dump([revs, W, word_idx_map, vocab, numclasses, W2], open("mr.p", "wb"))
    print "dataset created!"
    
