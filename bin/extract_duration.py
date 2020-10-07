# -*- coding: utf-8 -*-
# Copyright 2020 Minh Nguyen (@dathudeptrai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract durations based-on tacotron-2 alignments for FastSpeech."""
import tensorflow as tf
import sys
import argparse
import logging
import os
import numpy as np
import tensorflow_tts as tts
from tqdm import tqdm
from numba import jit
from tensorflow_tts.models import TFTacotron2
from tensorflow_tts.utils import return_strategy
from Processor import JSpeechProcessor

sys.path.append(".")

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

# return strategy
STRATEGY = return_strategy()

class Config(object):
    def __init__(self,outdir,vocab_size=149,n_speakers=1,batch_size=8):
        # tacotron2 params
        self.vocab_size = vocab_size                    # default
        self.embedding_hidden_size = 512            # 'embedding_hidden_size': 512
        self.initializer_range = 0.02                     # 'initializer_range': 0.02
        self.n_speakers = n_speakers                   # 'n_speakers': 1
        self.layer_norm_eps = 1e-6
        self.embedding_dropout_prob = 0.1          # 'embedding_dropout_prob': 0.1
        self.n_conv_encoder = 5                        # 'n_conv_encoder': 5
        self.encoder_conv_filters = 512                # 'encoder_conv_filters': 512
        self.encoder_conv_kernel_sizes = 5           # 'encoder_conv_kernel_sizes': 5
        self.encoder_conv_activation = 'relu'         # 'encoder_conv_activation': 'relu'
        self.encoder_conv_dropout_rate = 0.5        # 'encoder_conv_dropout_rate': 0.5
        self.encoder_lstm_units = 256                  # 'encoder_lstm_units': 256
        self.n_prenet_layers = 2                         # 'n_prenet_layers': 2
        self.prenet_units = 256                          # 'prenet_units': 256
        self.prenet_activation = 'relu'                   # 'prenet_activation': 'relu'
        self.prenet_dropout_rate = 0.5                  # 'prenet_dropout_rate': 0.5
        self.decoder_lstm_units = 1024                 # 'decoder_lstm_units': 1024
        self.n_lstm_decoder = 1                          # 'n_lstm_decoder': 1
        self.attention_type = 'lsa'                        # 'attention_type': 'lsa'
        self.attention_dim = 128                         # 'attention_dim': 128
        self.attention_filters = 32                        # 'attention_filters': 32
        self.attention_kernel = 31                       # 'attention_kernel': 31
        self.n_mels = 20                                   # 'n_mels': 80
        self.reduction_factor = 1                         # 'reduction_factor': 1
        self.n_conv_postnet = 5                          # 'n_conv_postnet': 5
        self.postnet_conv_filters = 512                  # 'postnet_conv_filters': 512
        self.postnet_conv_kernel_sizes = 5             # 'postnet_conv_kernel_sizes': 5
        self.postnet_dropout_rate = 0.1                # 'postnet_dropout_rate': 0.1
        
        # data
        self.batch_size = batch_size
        self.test_size = 0.05
        self.mel_length_threshold = 0
        self.guided_attention = 0.2
        
        # optimizer
        self.initial_learning_rate = 0.001
        self.end_learning_rate = 0.00001
        self.decay_steps = 150000
        self.warmup_proportion = 0.02
        self.weight_decay= 0.001
        
        # interval
        self.train_max_steps = 200000              
        self.save_interval_steps = 2000             
        self.eval_interval_steps = 500               
        self.log_interval_steps = 200                
        self.start_schedule_teacher_forcing = 200001
        self.start_ratio_value = 0.5               
        self.schedule_decay_steps = 50000     
        self.end_ratio_value = 0.0
        self.num_save_intermediate_results = 1
        
        self.outdir = outdir
        self.items = { 
            "outdir": outdir, 
            "batch_size": self.batch_size,
            "train_max_steps": self.train_max_steps,
            "log_interval_steps": self.log_interval_steps,
            "eval_interval_steps": self.eval_interval_steps,
            "save_interval_steps": self.save_interval_steps,
            "num_save_intermediate_results": self.num_save_intermediate_results 
        }
        
    def __getitem__(self, key):
        return self.items[key]
       
def generate_datasets(items, config, max_seq_length, max_mel_length):

    def _guided_attention(char_len, mel_len, max_char_len, max_mel_len, g=0.2):
        """Guided attention. Refer to page 3 on the paper."""
        max_char_seq = np.arange(max_char_len)
        max_char_seq = tf.expand_dims(max_char_seq, 0)  # [1, t_seq]
        # [mel_seq, max_t_seq]
        max_char_seq = tf.tile(max_char_seq, [max_mel_len, 1])

        max_mel_seq = np.arange(max_mel_len)
        max_mel_seq = tf.expand_dims(max_mel_seq, 1)  # [mel_seq, 1]
        # [mel_seq, max_t_seq]
        max_mel_seq = tf.tile(max_mel_seq, [1, max_char_len])

        right = tf.cast(max_mel_seq, tf.float32) / tf.constant(mel_len, dtype=tf.float32)
        left = tf.cast(max_char_seq, tf.float32) / tf.constant(char_len, dtype=tf.float32)

        ga_ = 1.0 - tf.math.exp(-((right - left) ** 2) / (2 * g * g))
        return tf.transpose(ga_[:mel_len, :char_len], (1, 0))
    
    def _generator():
        for item in items:
            tid, seq, feat_path, _ = item
            
            with open(feat_path, 'rb') as f:
                mel = np.fromfile(f, dtype='float32')
                mel = np.resize(mel, (-1, config.n_mels))
            
            seq_length = seq.shape[0]
            mel_length = mel.shape[0]
            if f is None or mel_length < config.mel_length_threshold:
                continue
                        
            # create guided attention (default).
            g_attention = _guided_attention(
                seq_length,
                mel_length,
                max_seq_length,
                max_mel_length,
                config.guided_attention
            )
            
            data = { 
                "utt_ids": tid,
                "input_ids": seq,
                "input_lengths": seq_length,
                "speaker_ids": 0,
                "mel_gts": mel,
                "mel_lengths": mel_length,
                "g_attentions": g_attention 
            }
                     
            yield data

    output_types = { 
        "utt_ids": tf.string, 
        "input_ids": tf.int32,
        "input_lengths": tf.int32,
        "speaker_ids": tf.int32,
        "mel_gts": tf.float32,
        "mel_lengths": tf.int32, 
        "g_attentions": tf.float32 
    }
                                                  
    datasets = tf.data.Dataset.from_generator(_generator, output_types=output_types)
            
    padding_values = {
        "utt_ids": " ",
        "input_ids": 0,
        "input_lengths": 0,
        "speaker_ids": 0,
        "mel_gts": 0.0,
        "mel_lengths": 0,
        "g_attentions": -1.0
    }
    
    padded_shapes = {
        "utt_ids": [],
        "input_ids": [None],
        "input_lengths": [],
        "speaker_ids": [],
        "mel_gts": [None, config.n_mels],      
        "mel_lengths": [],
        "g_attentions": [None, None]
    }
    
    datasets = datasets.padded_batch(
            config.batch_size * STRATEGY.num_replicas_in_sync, 
            padded_shapes=padded_shapes,
            padding_values=padding_values)
    datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
    
    return datasets
  
@jit(nopython=True)
def get_duration_from_alignment(alignment):
    D = np.array([0 for _ in range(np.shape(alignment)[0])])

    for i in range(np.shape(alignment)[1]):
        max_index = list(alignment[:, i]).index(alignment[:, i].max())
        D[max_index] = D[max_index] + 1
        
    return D
    
# python .\extract_duration.py --rootdir ./datasets/jsut/basic --outdir ./datasets/jsut/basic/durations --checkpoint model-211500.h5
def main():
    """Running extract tacotron-2 durations."""
    parser = argparse.ArgumentParser(description="Extract durations from charactor with trained Tacotron-2 ")
    parser.add_argument("--outdir", type=str, required=True, help="directory to save generated speech.")
    parser.add_argument("--rootdir", type=str, required=True, help="dataset directory root")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint file to be loaded.")
    parser.add_argument("--verbose",type=int,default=1,help="logging level. higher is more logging. (default=1)")
    parser.add_argument("--batch-size", default=8, type=int, help="batch size.")
    parser.add_argument("--win-front", default=2, type=int, help="win-front.")
    parser.add_argument("--win-back", default=2, type=int, help="win-front.")
    parser.add_argument("--use-window-mask", default=1, type=int, help="toggle window masking.")
    parser.add_argument("--save-alignment", default=0, type=int, help="save-alignment.")
    args = parser.parse_args()
    
    if args.checkpoint is not None and os.path.isdir(args.checkpoint):
        args.checkpoint = tf.train.latest_checkpoint(args.checkpoint)
        
    # set logger
    log_format = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,stream=sys.stdout,format=log_format)
    elif args.verbose > 0:
        logging.basicConfig(level=logging.INFO,stream=sys.stdout,format=log_format)
    else:
        logging.basicConfig(level=logging.WARN,stream=sys.stdout,format=log_format)
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    # select processor
    Processor = JSpeechProcessor
    
    processor = Processor(args.rootdir)     # for test
    config = Config(args.outdir, processor.vocab_size(),1, args.batch_size)
    
    max_seq_length = processor.max_seq_length()
    max_mel_length = processor.max_feat_length() // config.n_mels
    
    # generate datasets
    dataset = generate_datasets(processor.items, config, max_seq_length, max_mel_length)
    
    # define model.
    tacotron2 = TFTacotron2(config=config, training=True, name="tacotron2")
    
    #build
    input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    input_lengths = np.array([9])
    speaker_ids = np.array([0])
    mel_outputs = np.random.normal(size=(1, 50, config.n_mels)).astype(np.float32)
    mel_lengths = np.array([50])
    tacotron2(input_ids,input_lengths,speaker_ids,mel_outputs,mel_lengths,10,training=True)
    tacotron2.load_weights(args.checkpoint)
    tacotron2.summary()

    # apply tf.function for tacotron2.
    tacotron2 = tf.function(tacotron2, experimental_relax_shapes=True)
    
    for data in tqdm(dataset, desc="[Extract Duration]"):
        utt_ids = data["utt_ids"]
        input_lengths = data["input_lengths"]
        mel_lengths = data["mel_lengths"]
        utt_ids = utt_ids.numpy()
        real_mel_lengths = mel_lengths

        # tacotron2 inference.
        _, _, _, alignment_historys = tacotron2(
            **data,
            use_window_mask=args.use_window_mask,
            win_front=args.win_front,
            win_back=args.win_back,
            training=True,
        )

        # convert to numpy
        alignment_historys = alignment_historys.numpy()

        for i, alignment in enumerate(alignment_historys):
            real_char_length = input_lengths[i].numpy()
            real_mel_length = real_mel_lengths[i].numpy()
            alignment_mel_length = int(np.ceil(real_mel_length))
            alignment = alignment[:real_char_length, :alignment_mel_length]
            d = get_duration_from_alignment(alignment)  # [max_char_len]

            assert (np.sum(d) >= real_mel_length), f"{d}, {np.sum(d)}, {alignment_mel_length}, {real_mel_length}"
            if np.sum(d) > real_mel_length:
                rest = np.sum(d) - real_mel_length
                if d[-1] > rest:
                    d[-1] -= rest
                elif d[0] > rest:
                    d[0] -= rest
                else:
                    d[-1] -= rest // 2
                    d[0] -= rest - rest // 2

                assert d[-1] > 0 and d[0] > 0, f"{d}, {np.sum(d)}, {real_mel_length}"

            saved_name = utt_ids[i].decode("utf-8")

            # check a length compatible
            assert (len(d) == real_char_length), f"different between len_char and len_durations, {len(d)} and {real_char_length}"
            assert (np.sum(d) == real_mel_length), f"different between sum_durations and len_mel, {np.sum(d)} and {real_mel_length}"

            # save D to folder.
            d.astype(np.int32).tofile(os.path.join(args.outdir, f"{saved_name}.dur"))

            # save alignment to debug.
            if args.save_alignment == 1:
                figname = os.path.join(args.outdir, f"{saved_name}_alignment.png")
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                ax.set_title(f"Alignment of {saved_name}")
                im = ax.imshow(alignment, aspect="auto", origin="lower", interpolation="none")
                fig.colorbar(im, ax=ax)
                xlabel = "Decoder timestep"
                plt.xlabel(xlabel)
                plt.ylabel("Encoder timestep")
                plt.tight_layout()
                plt.savefig(figname)
                plt.close()
                
if __name__ == "__main__":
    main()