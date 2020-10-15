#!/usr/bin/python3
'''Copyright (c) 2018 Mozilla

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

# Train a LPCNet model (note not a Wavenet model)

import lpcnet
import sys
import numpy as np
import datetime
import h5py
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from ulaw import ulaw2lin, lin2ulaw

gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
if gpus:
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

nb_epochs = 120

# Try reducing batch_size if you run out of memory on your GPU
batch_size = 64

model, _, _ = lpcnet.new_lpcnet_model(training=True)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

feature_file = sys.argv[1]
pcm_file = sys.argv[2]     # 16 bit unsigned short PCM samples
frame_size = model.frame_size
nb_features = 55
nb_used_features = model.nb_used_features
feature_chunk_size = 15
pcm_chunk_size = frame_size*feature_chunk_size

# u for unquantised, load 16 bit PCM samples and convert to mu-law

data = np.fromfile(pcm_file, dtype='uint8')
nb_frames = len(data)//(4*pcm_chunk_size)

features = np.fromfile(feature_file, dtype='float32')

# limit to discrete number of frames
data = data[:nb_frames*4*pcm_chunk_size]
features = features[:nb_frames*feature_chunk_size*nb_features]

features = np.reshape(features, (nb_frames*feature_chunk_size, nb_features))

sig = np.reshape(data[0::4], (nb_frames, pcm_chunk_size, 1))
pred = np.reshape(data[1::4], (nb_frames, pcm_chunk_size, 1))
in_exc = np.reshape(data[2::4], (nb_frames, pcm_chunk_size, 1))
out_exc = np.reshape(data[3::4], (nb_frames, pcm_chunk_size, 1))
del data

print("ulaw std = ", np.std(out_exc))

features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))
features = features[:, :, :nb_used_features]
features[:,:,18:36] = 0

fpad1 = np.concatenate([features[0:1, 0:2, :], features[:-1, -2:, :]], axis=0)
fpad2 = np.concatenate([features[1:, :2, :], features[0:1, -2:, :]], axis=0)
features = np.concatenate([fpad1, features, fpad2], axis=1)


periods = (.1 + 50*features[:,:,36:37]+100).astype('int16')

in_data = np.concatenate([sig, pred, in_exc], axis=-1)

del sig
del pred
del in_exc

# dump models to disk as we go
checkpoint = ModelCheckpoint('lpcnet30_384_10_G16_{epoch:02d}.h5')

#Set this to True to adapt an existing model (e.g. on new data)
adaptation = False

if adaptation:
    #Adapting from an existing model
    model.load_weights('lpcnet24c_384_10_G16_120.h5')
    sparsify = lpcnet.Sparsify(0, 0, 1, (0.05, 0.05, 0.2))
    lr = 0.0001
    decay = 0
else:
    #Training from scratch
    sparsify = lpcnet.Sparsify(2000, 40000, 400, (0.05, 0.05, 0.2))
    lr = 0.001
    decay = 5e-5

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(optimizer=Adam(lr, amsgrad=True, decay=decay), loss='sparse_categorical_crossentropy')
model.save_weights('lpcnet30_384_10_G16_00.h5');
model.fit([in_data, features, periods], out_exc, batch_size=batch_size, epochs=nb_epochs, validation_split=0.0, callbacks=[checkpoint, sparsify, tensorboard_callback])
