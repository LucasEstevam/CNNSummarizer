import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import pdb
from conv_net_classes import LeNetConvPoolLayer
from conv_net_classes import MLPDropout
import os
this_dir, this_filename = os.path.split(__file__)

def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)


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

def clean_str(string, TREC=False):
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


def get_idx_from_sent(sent, word_idx_map, max_l=51, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x[:max_l+2*pad]

def make_idx_data_all(revs, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    revsin = []
    for rev in revs:
	if rev['num_words'] > 5:
            sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, filter_h)   
            sent.append(rev["y"])         
            test.append(sent)        
            train.append(sent)   
	    revsin.append(rev)
    train = np.array(train,dtype="int")
    test = np.array(test[1:100],dtype="int")
    return [train, test, revsin]     


   
if __name__=="__main__":
    mrppath = os.path.join(this_dir, "mr.p")
    x = cPickle.load(open(mrppath,"rb"))
    revs, W, word_idx_map, vocab, numclasses = x[0], x[1], x[2], x[3], x[4]            
    U = W
    classifierpath = os.path.join(this_dir, "classifier.save")
    savedparams = cPickle.load(open(classifierpath,'rb'))
    datasets = make_idx_data_all(revs, word_idx_map, max_l=70, k=300, filter_h=5)
    revsin = datasets[2]
    filter_hs=[3,4,5]
    conv_non_linear="relu"
    hidden_units=[100,numclasses]
    dropout_rate=[0.5]
    activations=[Iden]
    img_h = 70 + 4 + 4
    img_w = 300
    rng = np.random.RandomState(3435)
    batch_size=50
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))

#define model architecture
    x = T.matrix('x')   
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))                                  
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)    
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    classifier.params[0].set_value(savedparams[0])
    classifier.params[1].set_value(savedparams[1])
    k = 2
    for conv_layer in conv_layers:
        conv_layer.params[0].set_value( savedparams[k])
        conv_layer.params[1].set_value( savedparams[k+1])
        k = k + 2

    test_set_x = datasets[0][:,:img_h] 
    test_set_y = np.asarray(datasets[0][:,-1],"int32")



    test_pred_layers = []
    test_size = 1
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict_p(test_layer1_input)
    #test_error = T.mean(T.neq(test_y_pred, y))
    pdb.set_trace()
    model = theano.function([x],test_y_pred,allow_input_downcast=True)
    for i in range(0,test_set_x.shape[0]):
	rev = revsin[i]
	result = model(test_set_x[i,None])
    
