#!/usr/bin/python3
'''Copyright (c) 2017-2018 Mozilla

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import lpcnet
import sys
import struct
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.compat.v1.keras.layers import CuDNNGRU
from tensorflow.keras.layers import Layer, GRU, Dense, Conv1D, Embedding
import tensorflow.keras.backend as K

from ulaw import ulaw2lin, lin2ulaw
from mdense import MDense
import h5py
import re

max_rnn_neurons = 1
max_conv_inputs = 1
max_mdense_tmp = 1
  
Activations = {
    'LINEAR':0,
    'SIGMOID':1,
    'TANH':2,
    'RELU':3,
    'SOFTMAX':4
    }

def printVector(f, vector, name, dtype='float32'):
    print("name: {}, len: {}".format(name, len(vector)))
    v = np.reshape(vector, (-1))
    v = v.astype(dtype)
    f.write(struct.pack('I', len(v)))
    f.write(v.tobytes())

def printSparseVector(f, A, name):
    N = A.shape[0]
    W = np.zeros((0,))
    diag = np.concatenate([np.diag(A[:,:N]), np.diag(A[:,N:2*N]), np.diag(A[:,2*N:])])
    A[:,:N] = A[:,:N] - np.diag(np.diag(A[:,:N]))
    A[:,N:2*N] = A[:,N:2*N] - np.diag(np.diag(A[:,N:2*N]))
    A[:,2*N:] = A[:,2*N:] - np.diag(np.diag(A[:,2*N:]))
    printVector(f, diag, name + '_diag')
    idx = np.zeros((0,), dtype='int')
    for i in range(3*N//16):
        pos = idx.shape[0]
        idx = np.append(idx, -1)
        nb_nonzero = 0
        for j in range(N):
            if np.sum(np.abs(A[j, i*16:(i+1)*16])) > 1e-10:
                nb_nonzero = nb_nonzero + 1
                idx = np.append(idx, j)
                W = np.concatenate([W, A[j, i*16:(i+1)*16]])
        idx[pos] = nb_nonzero
    printVector(f, W, name)
    #idx = np.tile(np.concatenate([np.array([N]), np.arange(N)]), 3*N//16)
    printVector(f, idx, name + '_idx', dtype='int')

def dump_layer_ignore(self, f):
    print("ignoring layer " + self.name + " of type " + self.__class__.__name__)
    return False
Layer.dump_layer = dump_layer_ignore

def dump_sparse_gru(self, f):
    global max_rnn_neurons
    name = 'sparse_' + self.name
    print("printing layer " + name + " of type sparse " + self.__class__.__name__)
    
    weights = self.get_weights()
    printSparseVector(f, weights[1], name + '_recurrent_weights')
    
    v = np.reshape(weights, (-1))
    printVector(f, weights[-1], name + '_bias')
    if hasattr(self, 'activation'):
        activation = self.activation.__name__.upper()
    else:
        activation = 'TANH'
    if hasattr(self, 'reset_after') and not self.reset_after:
        reset_after = 0
    else:
        reset_after = 1
    neurons = weights[0].shape[1]//3
    max_rnn_neurons = max(max_rnn_neurons, neurons)      
    f.write(struct.pack('iii', weights[0].shape[1]//3, Activations[activation], reset_after))
    return True

def dump_gru_layer(self, f):
    global max_rnn_neurons
    name = self.name
    print("printing layer " + name + " of type " + self.__class__.__name__)
    weights = self.get_weights()
    printVector(f, weights[0], name + '_weights')
    printVector(f, weights[1], name + '_recurrent_weights')
    printVector(f, weights[-1], name + '_bias')
    if hasattr(self, 'activation'):
        activation = self.activation.__name__.upper()
    else:
        activation = 'TANH'
    if hasattr(self, 'reset_after') and not self.reset_after:
        reset_after = 0
    else:
        reset_after = 1
    neurons = weights[0].shape[1]//3
    max_rnn_neurons = max(max_rnn_neurons, neurons)
    f.write(struct.pack('iiii', weights[0].shape[0], weights[0].shape[1]//3, Activations[activation], reset_after))
    return True
CuDNNGRU.dump_layer = dump_gru_layer
GRU.dump_layer = dump_gru_layer

def dump_dense_layer_impl(name, weights, bias, activation, f):
    printVector(f, weights, name + '_weights')
    printVector(f, bias, name + '_bias')
    f.write(struct.pack('iii', weights.shape[0], weights.shape[1], Activations[activation]))

def dump_dense_layer(self, f):
    name = self.name
    print("printing layer " + name + " of type " + self.__class__.__name__)
    weights = self.get_weights()
    activation = self.activation.__name__.upper()
    dump_dense_layer_impl(name, weights[0], weights[1], activation, f)
    return False

Dense.dump_layer = dump_dense_layer

def dump_mdense_layer(self, f):
    global max_mdense_tmp
    name = self.name
    print("printing layer " + name + " of type " + self.__class__.__name__)
    weights = self.get_weights()
    printVector(f, np.transpose(weights[0], (1, 2, 0)), name + '_weights')
    printVector(f, np.transpose(weights[1], (1, 0)), name + '_bias')
    printVector(f, np.transpose(weights[2], (1, 0)), name + '_factor')
    activation = self.activation.__name__.upper()
    max_mdense_tmp = max(max_mdense_tmp, weights[0].shape[0]*weights[0].shape[2])
    f.write(struct.pack('iiii', weights[0].shape[1], weights[0].shape[0], weights[0].shape[2], Activations[activation]))
    return False
MDense.dump_layer = dump_mdense_layer

def dump_conv1d_layer(self, f):
    global max_conv_inputs
    name = self.name
    print("printing layer " + name + " of type " + self.__class__.__name__)
    weights = self.get_weights()
    printVector(f, weights[0], name + '_weights')
    printVector(f, weights[-1], name + '_bias')
    activation = self.activation.__name__.upper()
    max_conv_inputs = max(max_conv_inputs, weights[0].shape[1]*weights[0].shape[0])
    f.write(struct.pack('iiii', weights[0].shape[1], weights[0].shape[0], weights[0].shape[2], Activations[activation]))
    return True
Conv1D.dump_layer = dump_conv1d_layer


def dump_embedding_layer_impl(name, weights, f):
    printVector(f, weights, name + '_weights')
    f.write(struct.pack('ii', weights.shape[0], weights.shape[1]))

def dump_embedding_layer(self, f):
    name = self.name
    print("printing layer " + name + " of type " + self.__class__.__name__)
    weights = self.get_weights()[0]
    dump_embedding_layer_impl(name, weights, f)
    return False
Embedding.dump_layer = dump_embedding_layer

model, _, _ = lpcnet.new_lpcnet_model(rnn_units1=384, use_gpu=False)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

model.load_weights(sys.argv[1])

bf = open('nnet_data.bin', 'wb')

embed_size = lpcnet.embed_size

E = model.get_layer('embed_sig').get_weights()[0]
W = model.get_layer('gru_a').get_weights()[0][:embed_size,:]
dump_embedding_layer_impl('gru_a_embed_sig', np.dot(E, W), bf)
W = model.get_layer('gru_a').get_weights()[0][embed_size:2*embed_size,:]
dump_embedding_layer_impl('gru_a_embed_pred', np.dot(E, W), bf)
W = model.get_layer('gru_a').get_weights()[0][2*embed_size:3*embed_size,:]
dump_embedding_layer_impl('gru_a_embed_exc', np.dot(E, W), bf)
W = model.get_layer('gru_a').get_weights()[0][3*embed_size:,:]
#FIXME: dump only half the biases
b = model.get_layer('gru_a').get_weights()[2]
dump_dense_layer_impl('gru_a_dense_feature', W, b, 'LINEAR', bf)

layer_list = []
for i, layer in enumerate(model.layers):
    if layer.dump_layer(bf):
       layer_list.append(layer.name)

dump_sparse_gru(model.get_layer('gru_a'), bf)

bf.write(struct.pack('III', max_rnn_neurons, max_conv_inputs, max_mdense_tmp))

bf.close()
