import numpy as np
import tensorflow as tf
import sys
import argparse
import logging
import os
from tensorflow_tts.models import TFFastSpeech2

sys.path.append(".")

class Config(object):
    def __init__(self,outdir,n_speakers=1):
        # fastspeech2 params
        self.n_speakers = n_speakers
        self.encoder_hidden_size: 256
        self.encoder_num_hidden_layers: 3
        self.encoder_num_attention_heads: 2
        self.encoder_attention_head_size: 16  # in v1, = 384//2
        self.encoder_intermediate_size: 1024
        self.encoder_intermediate_kernel_size: 3
        self.encoder_hidden_act: "mish"
        self.decoder_hidden_size: 256
        self.decoder_num_hidden_layers: 3
        self.decoder_num_attention_heads: 2
        self.decoder_attention_head_size: 16  # in v1, = 384//2
        self.decoder_intermediate_size: 1024
        self.decoder_intermediate_kernel_size: 3
        self.decoder_hidden_act: "mish"
        self.variant_prediction_num_conv_layers: 2
        self.variant_predictor_filter: 256
        self.variant_predictor_kernel_size: 3
        self.variant_predictor_dropout_rate: 0.5
        self.num_mels: 20
        self.hidden_dropout_prob: 0.2
        self.attention_probs_dropout_prob: 0.1
        self.max_position_embeddings: 2048
        self.initializer_range: 0.02
        self.output_attentions: False
        self.output_hidden_states: False
    
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
    parser = argparse.ArgumentParser(description="Dump FastSpeech2")
    parser.add_argument("--outdir", default="./", type=str, help="directory to save pb or tflite file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint file to be loaded.")
    parser.add_argument("--tflite", type=bool, default=False,  help="saved model to tflite")
    args = parser.parse_args()
    
    # check directory existence(checkpoint)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        
    if args.checkpoint is not None and os.path.isdir(args.checkpoint):
        args.checkpoint = tf.train.latest_checkpoint(args.checkpoint)
    
    save_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    config = Config(args.outdir)
    
    # define model.
    fastspeech2 = TFFastSpeech2(config=config, training=False, name="fastspeech2", enable_tflite_convertible=args.tflite)
    
    #build       
    if args.tflite is True:
        print("dump tflite => vocab_size: {}".format(args.vocab_size))
        input_ids = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], tf.int32)
        speaker_ids = tf.convert_to_tensor([0], tf.int32)
        duration_gts = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], tf.int32)
        f0_gts = tf.convert_to_tensor([[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]], tf.float32)
        energy_gts = tf.convert_to_tensor([[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]], tf.float32)
        fastspeech2(input_ids,speaker_ids,duration_gts,f0_gts,energy_gts)
        fastspeech2.load_weights(args.checkpoint)
        fastspeech2.summary()
        fastspeech2_concrete_function = fastspeech2.inference_tflite.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([fastspeech2_concrete_function])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
    
        with open(os.path.join(args.outdir, "{}.tflite".format(save_name)), 'wb') as f:
            f.write(tflite_model)
        
    else:
        print("dump => vocab_size: {}".format(args.vocab_size))
        # tensorflow-gpu==2.3.0 bug to load_weight after call inference
        fastspeech2.inference(input_ids=tf.expand_dims(tf.convert_to_tensor([1], dtype=tf.int32), 0),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
            speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
            energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32))
        
        fastspeech2.load_weights(args.checkpoint)   
        tf.saved_model.save(fastspeech2, os.path.join(args.outdir, save_name), signatures = fastspeech2.inference)
    
if __name__ == "__main__":
    main()