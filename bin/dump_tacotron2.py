import numpy as np
import tensorflow as tf
import sys
import argparse
import logging
import os
from tensorflow_tts.models import TFTacotron2
from Processor import JSpeechProcessor

sys.path.append(".")

class Config(object):
    def __init__(self,outdir,vocab_size=65535,n_speakers=1):
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
        self.batch_size = 32
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
        
def main():
    parser = argparse.ArgumentParser(description="Dump Tacotron2")
    parser.add_argument("--outdir", default="./", type=str, help="directory to save pb or tflite file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint file to be loaded.")
    parser.add_argument("--vocab_size", type=int, required=True, help="vocab size")
    parser.add_argument("--tflite", type=bool, default=True,  help="saved model to tflite")
    args = parser.parse_args()
    
    # check directory existence(checkpoint)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        
    if args.checkpoint is not None and os.path.isdir(args.checkpoint):
        args.checkpoint = tf.train.latest_checkpoint(args.checkpoint)
        
    config = Config(args.outdir, args.vocab_size)
    
    # define model.
    tacotron2 = TFTacotron2(config=config, training=False, name="tacotron2", enable_tflite_convertible=args.tflite)
    
    # Newly added :
    tacotron2.setup_window(win_front=6, win_back=6)
    tacotron2.setup_maximum_iterations(3000)
    
    #build       
    input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    input_lengths = np.array([9])
    speaker_ids = np.array([0])
    mel_outputs = np.random.normal(size=(1, 50, config.n_mels)).astype(np.float32)
    mel_lengths = np.array([50])
    tacotron2(input_ids,input_lengths,speaker_ids,mel_outputs,mel_lengths,10,training=False)
    tacotron2.summary()
    tacotron2.load_weights(args.checkpoint)
        
    if args.tflite:
        tacotron2_concrete_function = tacotron2.inference_tflite.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([tacotron2_concrete_function])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
    
        # Save the TF Lite model.
        with open(os.path.join(args.outdir, 'tacotron2.tflite'), 'wb') as f:
            f.write(tflite_model)

        print('Model size is %f MBs.' % (len(tflite_model) / 1024 / 1024.0) )
    else:
        inference_inputs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        _, _, _, _ = tacotron2.inference(
        input_ids = tf.expand_dims(tf.convert_to_tensor(inference_inputs, dtype=tf.int32), 0),
        input_lengths = tf.convert_to_tensor([len(inference_inputs)], tf.int32),
        speaker_ids = tf.convert_to_tensor([0], dtype=tf.int32))
        tf.saved_model.save(tacotron2, os.path.join(args.outdir, 'test_saved'), signatures=tacotron2.inference)
    
if __name__ == "__main__":
    main()