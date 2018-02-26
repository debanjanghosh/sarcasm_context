# using the following for reference:
# https://github.com/umass-semeval/semeval16/blob/master/semeval/lstm_words.py 
# Also Chris's work on persuasion for diferent custom layers of Attention
#TODO - add a LIWC vector with the sentence vector
import os
os.environ["THEANO_FLAGS"] = "floatX=float32"
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import pandas as pd
import logging
import math
import pickle
import os
import timeit
import time
import lasagne
from lasagne.layers import get_output_shape
from lasagne.regularization import apply_penalty, l2

from com.ccls.lstm.preprocess.utils import str_to_bool
from com.ccls.lstm.main.hidey_layers import AttentionWordLayer, 
AttentionSentenceLayer, WeightedAverageWordLayer, WeightedAverageSentenceLayer


class SarcasmLstmAttentionSeparate:
    def __init__(self, 
                W=None, 
                W_path=None, 
                K=300, 
                num_hidden=256,
                batch_size=None,
                grad_clip=100., 
                max_sent_len=200, 
                num_classes=2, 
                **kwargs):
        
        W = W
        V = len(W)
        K = int(K)
        num_hidden = int(num_hidden)
        batch_size = int(batch_size)
        grad_clip = int(grad_clip)
        max_seq_len = int(max_sent_len)
        max_post_len = int(kwargs["max_post_len"])
        max_len = max(max_seq_len, max_post_len)

        max_seq_len = max_len
        max_post_len = max_len

        num_classes = int(num_classes)    
        
        ''' Boolean on and/or sentence and word level attention'''
        ''' to use context or not'''
        
        separate_attention_context_sents = str_to_bool(kwargs["separate_attention_context"])
        separate_attention_response_sents = str_to_bool(kwargs["separate_attention_response"])
        
        separate_attention_context_words = str_to_bool(kwargs["separate_attention_context_words"])
        separate_attention_response_words = str_to_bool(kwargs["separate_attention_response_words"])

        print("separate_attention_context_sentence is : {}\n".format(separate_attention_context_sents))
        print("separate_attention_response_sentence is : {}\n".format(separate_attention_response_sents))
        print("separate_attention_context_words is : {}\n".format(separate_attention_context_words))
        print("separate_attention_response_words is : {}\n".format(separate_attention_response_words))

        #B x S x N tensor of batches of context
        idxs_context = T.itensor3('idxs_context') #imatrix, i = int
        #B x S x N matrix
        mask_context_words = T.itensor3('mask_context_words')
        #B x S matrix
        mask_context_sents = T.imatrix('mask_context_sents')

        #B x S x N tensor of batches of responses
        idxs_response = T.itensor3('idxs_response') #imatrix, i = int
        
        #B x S X N matrix for words
        mask_response_words = T.itensor3('mask_response_words')
        #B x S matrix for sentences
        mask_response_sents = T.imatrix('mask_response_sents')
        
        #B-long vector
        gold = T.ivector('y')
        
        # dropout
        dropout_val = T.scalar('p_dropout')
        
        #lambda, cost
        lambda_cost = T.scalar('lambda_w')
        
        #biases
        biases_cost = T.matrix('biases')
        
        #weights
        weights = T.ivector('weights')
        
        ''' check biases'''
        biases_present = False
        if biases_present:
            lstm_biases = lasagne.layers.InputLayer(shape=(None,1),
                                                  input_var=biases_cost)
            
        ''' building the context  layer via function'''
        if separate_attention_context_sents:
            lstm_hidden_context,lstm_attn_words_context,lstm_attn_sents_context = self.buildThePostLayer(idxs_context,mask_context_words,\
                                                mask_context_sents,\
                                                separate_attention_context_words,\
                                                separate_attention_context_sents,num_hidden,grad_clip,V,K,W,max_post_len,max_sent_len)
        

        ''' do the same for response layer'''
        if separate_attention_response_sents:
            lstm_hidden_response,lstm_attn_words_response,lstm_attn_sents_response = self.buildThePostLayer(idxs_response,mask_response_words,\
                                                mask_context_sents,\
                                                separate_attention_context_words,\
                                                separate_attention_context_sents,num_hidden,grad_clip,V,K,W,max_post_len,max_sent_len)
        
        print('...')
        print('finished compiling...')
        
        ''' prepare the final network of connections now'''
        if separate_attention_response_sents and separate_attention_context_sents:
            output,network = self.buildNetwork(lstm_hidden_context,lstm_hidden_response,num_classes)
        
        elif   separate_attention_context_sents:
            output,network = self.buildNetworkOnlyContext(lstm_hidden_context,num_classes)

        
        '''Define objective function (cost) to minimize mean cross-entropy error'''
        params = lasagne.layers.get_all_params(network)
        cost = lasagne.objectives.categorical_crossentropy(output, gold).mean()
        lambda_w = .000001
        cost += lambda_w*apply_penalty(params, l2)
        grad_updates = lasagne.updates.adam(cost, params)

        test_output = lasagne.layers.get_output(network, deterministic=True)
        val_cost_fn = lasagne.objectives.categorical_crossentropy(
            test_output, gold).mean()
        preds = T.argmax(test_output, axis=1)

        val_acc_fn = T.mean(T.eq(preds, gold),
                            dtype=theano.config.floatX)
        
        if separate_attention_context_sents and separate_attention_response_sents:
            
            self.val_fn = theano.function([idxs_context, mask_context_words, mask_context_sents, idxs_response, \
                                           mask_response_words, mask_response_sents, gold], [val_cost_fn, val_acc_fn, preds],
                                          allow_input_downcast=True,on_unused_input='warn')
            # Compile train objective
            print "Compiling training, testing, prediction functions"
            self.train = theano.function(inputs = [idxs_context, mask_context_words, mask_context_sents,\
                                                   idxs_response, mask_response_words, mask_response_sents, gold],\
                                          outputs = cost, updates = grad_updates, allow_input_downcast=True,on_unused_input='warn')
            
            self.test = theano.function(inputs = [idxs_context, mask_context_words, mask_context_sents,idxs_response,\
                                                  mask_response_words, mask_response_sents, gold],\
                                                   outputs = val_acc_fn,allow_input_downcast=True,on_unused_input='warn')
            
            self.pred = theano.function(inputs = [idxs_context, mask_context_words, mask_context_sents, \
                                                  idxs_response, mask_response_words, mask_response_sents],\
                                        outputs = preds,allow_input_downcast=True,on_unused_input='warn')
            
        elif separate_attention_context_sents:
            
             self.val_fn = theano.function([idxs_context, mask_context_words, mask_context_sents,  \
                                            gold], [val_cost_fn, val_acc_fn, preds],
                                          allow_input_downcast=True,on_unused_input='warn')
             
             print "Compiling training, testing, prediction functions"
             self.train = theano.function(inputs = [idxs_context, mask_context_words, mask_context_sents,\
                                                    gold],\
                                          outputs = cost, updates = grad_updates, allow_input_downcast=True,on_unused_input='warn')
            
             self.test = theano.function(inputs = [idxs_context, mask_context_words, mask_context_sents,\
                                                   gold],\
                                                   outputs = val_acc_fn,allow_input_downcast=True,on_unused_input='warn')
            
             self.pred = theano.function(inputs = [idxs_context, mask_context_words, mask_context_sents \
                                                  ],\
                                        outputs = preds,allow_input_downcast=True,on_unused_input='warn')
            
            
        if separate_attention_response_sents:
            sentence_attention = lasagne.layers.get_output(lstm_attn_sents_response)
            #if add_biases:
            #    inputs = inputs[:-1]
            self.sentence_attention_response = theano.function([idxs_context, mask_context_words,\
                                                mask_context_sents,idxs_response, mask_response_words, mask_response_sents],
                                                      sentence_attention,
                                                      allow_input_downcast=True,
                                                      on_unused_input='warn')
        if separate_attention_context_sents:
            sentence_attention_context = lasagne.layers.get_output(lstm_attn_sents_context)
            #if add_biases:
            #    inputs = inputs[:-1]
            self.sentence_attention_context = theano.function([idxs_context, mask_context_words,\
                                             mask_context_sents,idxs_response, mask_response_words, mask_response_sents],
                                                      [sentence_attention_context, preds],
                                                      allow_input_downcast=True,
                                                      on_unused_input='warn')
            
        if separate_attention_response_words:
            sentence_attention_words = lasagne.layers.get_output(lstm_attn_words_response)
            #if add_biases:
            #    inputs = inputs[:-1]
            self.sentence_attention_response_words = theano.function([idxs_context, mask_context_words, mask_context_sents,idxs_response, mask_response_words, mask_response_sents],
                                                      sentence_attention_words,
                                                      allow_input_downcast=True,
                                                      on_unused_input='warn')
        if separate_attention_context_words:
            sentence_attention_context_words = lasagne.layers.get_output(lstm_attn_words_context)
            #if add_biases:
            #    inputs = inputs[:-1]
            self.sentence_attention_context_words = theano.function([idxs_context, mask_context_words, mask_context_sents,idxs_response, mask_response_words, mask_response_sents],
                                                      sentence_attention_context_words,
                                                      allow_input_downcast=True,
                                                      on_unused_input='warn')

        '''compare the results with regular code and then add the bias etc. '''

    
    def buildNetwork(self,lstm_hidden_context,lstm_hidden_response,num_classes):
        
        lstm_concat = lasagne.layers.ConcatLayer([lstm_hidden_context,lstm_hidden_response])
        lstm_concat = lasagne.layers.DropoutLayer(lstm_concat,p=0.5)
        network = lasagne.layers.DenseLayer(
            lstm_concat,
            num_units=num_classes,
            nonlinearity=lasagne.nonlinearities.softmax
        )
        self.network = network
        return lasagne.layers.get_output(network),network

    def buildNetworkOnlyContext(self,lstm_hidden_context,num_classes):
        
        lstm_concat = lasagne.layers.ConcatLayer([lstm_hidden_context])
        lstm_concat = lasagne.layers.DropoutLayer(lstm_concat,p=0.5)
        network = lasagne.layers.DenseLayer(
            lstm_concat,
            num_units=num_classes,
            nonlinearity=lasagne.nonlinearities.softmax
        )
        self.network = network
        return lasagne.layers.get_output(network),network

        
    def buildThePostLayer(self,idxs_type,mask_type_words,mask_type_sents,separate_attention_words,\
                          separate_attention_sentences,num_hidden,grad_clip,V,K,W,max_post_len,max_sent_len):
        
         
        ''' now build the layers for context and response'''
        ''' Inputlayer - the input from data'''
        ''' for idxs, mask words, mask sentences'''
        
        lstm_idxs_type = lasagne.layers.InputLayer(shape=(None,max_post_len,max_sent_len),
                                                      input_var = idxs_type)
        lstm_mask_sents = lasagne.layers.InputLayer(shape=(None,max_post_len), input_var = mask_type_sents)
        lstm_mask_words =  lasagne.layers.InputLayer(shape=(None,max_post_len,max_sent_len),
                                                      input_var = mask_type_words)
 
        
        ''' now add the embedding layer '''
        ''' word embedding layer of V * K ? '''
        #now B x S x N x D
        lstm_embed_type = lasagne.layers.EmbeddingLayer(lstm_idxs_type,input_size=V,output_size=K,W=W)
        #need to know about the following statememnt
        lstm_embed_type.params[lstm_embed_type.W].remove('trainable')
        
        ''' check whether word level attention exist or not '''
        lstm_attention_words = None
        lstm_attention_sents = None
        
        if separate_attention_words:
            lstm_attention_words = AttentionWordLayer([lstm_embed_type,lstm_mask_words],K)
            lstm_avg_words_context = WeightedAverageWordLayer([lstm_embed_type,lstm_attention_words])
            lstm_avg_context = lstm_avg_words_context
        else:
            lstm_avg_words_context = WeightedAverageWordLayer([lstm_embed_type,lstm_mask_words])
            lstm_avg_context = lstm_avg_words_context

        '''build the lstm layer '''
        ''' not sure about the masking context sent as mask_input'''
        lstm_layer_context = lasagne.layers.LSTMLayer(lstm_avg_context,num_hidden,
                                               nonlinearity=lasagne.nonlinearities.tanh,
                                               grad_clipping=grad_clip,
                                               mask_input=lstm_mask_sents)
        
        ''' check whether sentence level attention exist or not '''
        ''' sentence layer is on top of word layer, so the first input of attention is coming from words '''
        if separate_attention_sentences:
            print("separate attention context\n")
            lstm_attention_sents = AttentionSentenceLayer([lstm_layer_context, lstm_mask_sents], num_hidden)        
            lstm_avg_sents_context = WeightedAverageSentenceLayer([lstm_layer_context, lstm_attention_sents])
            print(" attention weighted average sentence layer shape: {}\n".format(get_output_shape(lstm_avg_sents_context)))
        else:
            print("just averaged context without attention\n")
            lstm_avg_sents_context = WeightedAverageSentenceLayer([lstm_layer_context, lstm_mask_sents])
            print(" attention weighted average sentence layer shape: {}\n".format(get_output_shape(lstm_avg_sents_context)))
    
        return lstm_avg_sents_context,lstm_attention_words,lstm_attention_sents
    
    def get_params(self):
        return lasagne.layers.get_all_param_values(self.network)

    def set_params(self, params):
        lasagne.layers.set_all_param_values(self.network, params)

    def save(self, filename):
        params = self.get_params()
        np.savez_compressed(filename, *params)

def load(model, filename):
    params = np.load(filename)
    param_keys = map(lambda x: 'arr_' + str(x), sorted([int(i[4:]) for i in params.keys()]))
    param_values = [params[i] for i in param_keys]
    lasagne.layers.set_all_param_values(model.network, param_values)