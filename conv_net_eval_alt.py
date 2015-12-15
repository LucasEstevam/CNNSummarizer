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
	else:
	    x.append(word_idx_map["UUUKKK"])
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
    revs, W, word_idx_map, vocab, numclasses, W2 = x[0], x[1], x[2], x[3], x[4] , x[5]
    U = W
    classifierpath = os.path.join(this_dir, "classifier.save")
    savedparams = cPickle.load(open(classifierpath,'rb'))
    mode2 = savedparams[1]
    savedparams = savedparams[0]
    if mode2 == "-rand":
        hunits = 10
	dims = 25
        U = W2
    else:
	hunits = 100
	dims = 300
	U = W
    datasets = make_idx_data_all(revs, word_idx_map, max_l=70, k=dims, filter_h=5)
    revsin = datasets[2]
    filter_hs=[3,4,5]
    conv_non_linear="relu"
    hidden_units=[hunits,numclasses]
    dropout_rate=[0.5]
    activations=[Iden]
    img_h = 70 + 4 + 4
    img_w = dims
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
    currentClass = 0
    currentBestScore = 0
    currentBestSentence = ""
    for i in range(0,test_set_x.shape[0]):
	rev = revsin[i]
	result = model(test_set_x[i,None])
	scoresum = result[0][currentClass]
	if rev['y'] > currentClass:
	    currentClass = rev['y']
	    scoresum = result[0][currentClass]
   	    currentBestScore = 0
	    print "best for class" + str(currentClass)
	    print currentBestSentence
 	    currentBestSentence = ""
	if scoresum > currentBestScore:
	    currentBestScore = scoresum
    	    currentBestSentence = rev
    print "best for class" + str(currentClass)
    print currentBestSentence
    
