# using the following for reference:
# https://github.com/umass-semeval/semeval16/blob/master/semeval/lstm_words.py 
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import pandas as pd
import nltk
import logging
import math
import pickle
import os
import timeit
import time
import lasagne
from lasagne.layers import get_output_shape
from com.ccls.lstm.preprocess.process_properties import PreProcessor
from com.ccls.lstm.preprocess.utils import str_to_bool 

'''
Preparing input data for the LSTM operations 
'''


def load_twitter_wsd_data(processor):
    
    return_dict = processor.__dict__
    path = processor.output
    both = str_to_bool(processor.both)
    attention = str_to_bool(processor.attention)
    separate = str_to_bool(processor.separate)
    batch_size = int(processor.batch_size)
    return_dict["lstm"] = processor.lstm
    data_type = processor.data_type
    print("The path to the input is: {}\n".format(path))
    target = processor.target
    
    ''' Running different LSTM for context/reply posts '''
    ''' (by default this code assuming sepeare LSTM for posts)'''
    
    if separate == True:
        train_file = path   + 'tweet.' + target + '.target.double.TRAIN' + '.pkl'
        test_file = path +    'tweet.' + target + '.target.double.TEST' + '.pkl'
    else:
        train_file = path   + 'tweet.' + target + '.target.single.TRAIN' + '.pkl'
        test_file = path +    'tweet.' + target + '.target.single.TEST' + '.pkl'
        
    max_l = 200

    print "loading data...",
    #logger.error("loading data...");

    print("the train_file is: {}\n".format(train_file))
    x = cPickle.load(open(train_file,"rb"))
    if separate == True:
        train_data, W, word_idx_map, max_sent_len,max_sent_len_2 = x[0], x[1], x[2], x[3], x[4]
    else:
        train_data, W, word_idx_map, max_sent_len = x[0], x[1], x[2], x[3]
        
    return_dict["W"] = W
    
    ''' # of sentences in a post (tweet, Reddit post)'''
    max_sent_len = 50 
    max_post_len = int(processor.max_sentences_response)
    max_context_len = int(processor.max_sentences_context)

    #if separate == True:
    #    max_post_len = 30
    print("max post len: {}\n".format(max_post_len))

    return_dict["max_sent_len_basic"] = max_l
    return_dict["max_sent_len"] = max_sent_len
    return_dict["max_post_len"] = max_post_len
    print("this is the test file: {}\n".format(test_file))
    test_data = cPickle.load(open(test_file,'rb'))
 #   test_post_data = cPickle.load(open(test_post_file,'rb'))

    ''' by default we are expecting separate LSTMs for context/post/replies'''
    
    print("This is separate loading\n")

    X_train_indx_context,X_train_indx_response,y_train =  \
    text_to_indx_sentence_separate(train_data, word_idx_map, max_context_len)
    X_test_indx_context, X_test_indx_response, y_test = \
    text_to_indx_sentence_separate(test_data, word_idx_map, max_context_len)
    print(X_test_indx_context[0])


    X_train_indx_pad_context, X_train_indices_mask_sents_context, \
    X_train_indices_mask_posts_context = text_to_indx_mask(X_train_indx_context, max_sent_len, max_context_len)
        
    X_train_indx_pad_response, X_train_indices_mask_sents_response, \
    X_train_indices_mask_posts_response = text_to_indx_mask(X_train_indx_response, max_sent_len, max_post_len)
        
    print(X_train_indx_pad_context.shape)
    print(X_train_indx_pad_response.shape)

    X_test_indx_pad_context, X_test_indices_mask_sents_context, X_test_indices_mask_posts_context = \
    text_to_indx_mask(X_test_indx_context, max_sent_len, max_context_len)
    X_test_indx_pad_response, X_test_indices_mask_sents_response, X_test_indices_mask_posts_response = \
    text_to_indx_mask(X_test_indx_response, max_sent_len, max_post_len)

    training = X_train_indx_pad_context, X_train_indices_mask_sents_context, \
    X_train_indices_mask_posts_context,X_train_indx_pad_response, X_train_indices_mask_sents_response, X_train_indices_mask_posts_response
    train_set_y = np.asarray(y_train)

    testing = X_test_indx_pad_context, X_test_indices_mask_sents_context, \
    X_test_indices_mask_posts_context,X_test_indx_pad_response, X_test_indices_mask_sents_response, X_test_indices_mask_posts_response
    test_set_y = np.asarray(y_test)

    
    return training,train_set_y,testing,test_set_y,return_dict


'''

Following are a suite of helper functions for masking and retrieving index of words
'''

def make_mask(post_length, max_post_len):
    return [1]*min(max_post_len, post_length) + [0]*max(0,max_post_len-post_length)

def make_sentence_mask(post_length, max_post_len, sentence_lengths, max_sent_len):
    ret = []
    for i in range(min(post_length, max_post_len)):
        ret.append([1]*min(sentence_lengths[i], max_sent_len) + [0]*max(0,max_sent_len - sentence_lengths[i]))
    for i in range(max_post_len - post_length):
        ret.append([0]*max_sent_len)
    return ret    

def pad_post(X_train_indx, max_l, max_s):
    padded_post = []
    for sentence in X_train_indx:
        padded_post.append(sentence[:max_l] + [0]*max(0, max_l-len(sentence)))
    return padded_post[:max_s] + [[0]*max_l]*max(0,(max_s-len(padded_post)))

def text_to_indx(train_data, word_idx_map, max_l):
    X = []
    y = []
    for query in train_data:
        #text = query["text"].split()
        text = nltk.word_tokenize(query["text"])
        text = text[:max_l]
        y_val = query["y"]
        out = []
        for word in text:
            if word in word_idx_map:
                out.append(word_idx_map[word])
            else:
                # unknown word
                out.append(1)
        X.append(out)
        y.append(y_val)
    return X,y



def text_to_indx_sentence_separate(train_data, word_idx_map, max_context_len):
    X = []
    X_context = []
    y = []
    
    ''' for tweets we split on "posts" and not on periods. posts are separated by '|||' '''
    ''' e.g. for a diologue on tweets, all previous replied tweets are chronologically 
        added using |||'''
    
    max_sentence_length = 50
    for query in train_data:
        context = query["x1"]
        response = query["x2"]
        y_val = query["y"]
        sentences_arr = []
        sentences = context.split(' ||| ')
        if len(sentences) > max_context_len:
            sentences = sentences[-max_context_len:]
        for sentence in sentences:
            out = []
            ''' if we have the marker(|||) in the sentence, now remove it'''
            sentence = sentence.replace('|||','')
           # words = nltk.word_tokenize(sentence)
            words = sentence.split()
            words = words[:max_sentence_length]
            for word in words:
                if word in word_idx_map:
                    out.append(word_idx_map[word])
                else:
                    out.append(1)
            sentences_arr.append(out)   
        X_context.append(sentences_arr)
        sentences_arr = []
        try:
            response = response.encode('utf-8').strip().replace('\n',' ').replace('\r',' ').replace('\t',' ')
        except:
            response = response.decode('latin').strip().replace('\n',' ').replace('\r',' ').replace('\t',' ')

        sentences = nltk.sent_tokenize(response)
        # TODO, how to deal with long context
        #if len(sentences) > max_post_length:
        #    continue
        for sentence in sentences:
            out = []
      #      words = nltk.word_tokenize(sentence)
            words = sentence.split()
            words = words[:max_sentence_length]
            for word in words:
                if word in word_idx_map:
                    out.append(word_idx_map[word])
                else:
                    out.append(1)
            sentences_arr.append(out)   

        X.append(sentences_arr)
        y.append(y_val)
    return X_context, X, y


def pad_mask(X_train_indx, max_l):

    N = len(X_train_indx)
    X = np.zeros((N, max_l), dtype=np.int32)
    X_mask = np.zeros((N,max_l), dtype = np.int32)
    for i, x in enumerate(X_train_indx):
        n = len(x)
        if n < max_l:
            X[i, :n] = x
            X_mask[i, :n] = 1
        else:
            X[i, :] = x[:max_l]
            X_mask[i, :] = 1

    return X,X_mask

def text_to_indx_mask(X_train_indx, max_sent_len, max_post_len):
    X_train_indx_pad = []
    indices_mask_sents = []
    indices_mask_posts = []

    for post in X_train_indx:
        post_length = len(post)
        sentences_length = [len(x) for x in post]
        curr_indx = pad_post(post, max_sent_len, max_post_len)
        curr_mask_sents = make_sentence_mask(post_length, max_post_len, sentences_length, max_sent_len)
        curr_mask_posts = make_mask(post_length, max_post_len)


        X_train_indx_pad.append(curr_indx)
        indices_mask_sents.append(curr_mask_sents)
        indices_mask_posts.append(curr_mask_posts)

    return np.asarray(X_train_indx_pad, dtype=np.int32), np.asarray(indices_mask_sents,dtype=np.int32), np.asarray(indices_mask_posts, dtype=np.int32)


def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

def shared_dataset_mask(data_x,data_y, data_z, borrow=True):
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """

        shared_x = theano.shared(np.asarray(data_x,
                                                 dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_z = theano.shared(np.asarray(data_z,
                                                 dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue

        return shared_x, shared_y, T.cast(shared_z, 'int32')

def iterate_minibatches(inputs,inputs2, targets, batch_size, shuffle=False):
    ''' Taken from the mnist.py example of Lasagne'''

    targets = np.asarray(targets)
    assert inputs.shape[0] == targets.size
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], inputs2[excerpt], targets[excerpt]

def split_train_test(X_tuple, y, train_size=.9, random_state=123):
    np.random.seed(random_state)
    tuple_len = len(X_tuple)
    x_shape = X_tuple[0].shape[0]
    #x_shape = X_tuple[0].eval().shape[0]
    train_len= int(math.floor(x_shape * train_size))
    #print("train_len: {}\n".format(train_len))
    indxs = np.random.choice(x_shape, x_shape, False)

    x_train = []
    x_holdout = []
    for i in range(tuple_len):
        #test = X_tuple[i][indxs[0:train_len]]
        #print("shape of test: {}\n".format(test.eval().shape))
        x_train.append(X_tuple[i][indxs[0:train_len]])
        x_holdout.append(X_tuple[i][indxs[train_len:]])

    #print(y)
    #print(y.shape)
    y_train = y[indxs[0:train_len]]
    y_holdout = y[indxs[train_len:]]

    return x_train, x_holdout, y_train, y_holdout 

def get_batch(X, y, batch_idxs):
    x_arr = []
    for i in range(len(X)):
        x_arr.append(X[i][batch_idxs]) # .eval())
    y = y[batch_idxs] # .eval()

    return x_arr, y

def read_model_data(model, filename):
    """Unpickles and loads parameters into a Lasagne model."""
    filename = os.path.join('./', '%s.%s' % (filename, 'params'))
    with open(filename, 'r') as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(model, data)


def write_model_data(model, filename):
    """Pickels the parameters within a Lasagne model."""
    data = lasagne.layers.get_all_param_values(model)
    filename = os.path.join('./', filename)
    filename = '%s.%s' % (filename, 'params')
    with open(filename, 'w+') as f:
        pickle.dump(data, f)

def text_to_indx_sentence(train_data, word_idx_map, max_post_length):
    X = []
    y = []
    ''' always check the type of text; depending upon that we need split or tokenizer function'''
    max_sentence_length = 50
    for query in train_data:
        text = query["text"]
        y_val = query["y"]
        sentences_arr = []
        try:
            text = text.encode('utf-8').strip().replace('\n',' ').replace('\r',' ').replace('\t',' ')
        except:
            text = text.decode('latin').strip().replace('\n',' ').replace('\r',' ').replace('\t',' ')

     #   sentences = nltk.sent_tokenize(text.decode('utf-8'))
        sentences = nltk.sent_tokenize(text)
        
        for sentence in sentences:
            out = []
           # words = nltk.word_tokenize(sentence)
            words = sentence.split()
            words = words[:max_sentence_length]
            for word in words:
                if word in word_idx_map:
                    out.append(word_idx_map[word])
                else:
                    out.append(1)
            sentences_arr.append(out)

        X.append(sentences_arr)
        y.append(y_val)
    return X,y

