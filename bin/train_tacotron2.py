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
"""Train Tacotron2."""
import tensorflow as tf
import sys
import argparse
import logging
import os
import numpy as np
import yaml
import tensorflow_tts as tts
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow_tts.configs.tacotron2 import Tacotron2Config
from tensorflow_tts.models import TFTacotron2
from tensorflow_tts.optimizers import AdamWeightDecay, WarmUp
from tensorflow_tts.trainers import Seq2SeqBasedTrainer
from tensorflow_tts.utils import (calculate_2d_loss, calculate_3d_loss, return_strategy)
from Processor import JSpeechProcessor

sys.path.append(".")

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

# return strategy
STRATEGY = return_strategy()

"""
datasets
    feats
        one.f32
        two.f32
    metadata.csv    
"""

class Config(object):
    def __init__(self,outdir,vocab_size=149,n_speakers=1):
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
        
        self.outdir = outdir
        self.items = { "outdir": outdir, 
                         "batch_size": self.batch_size,
                         "train_max_steps": self.train_max_steps,
                         "log_interval_steps": self.log_interval_steps,
                         "eval_interval_steps": self.eval_interval_steps,
                         "save_interval_steps": self.save_interval_steps }
        
    def __getitem__(self, key):
        return self.items[key]

def generate_datasets(items, config, max_mel_length, max_ids_length):

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
            text_ids, feat_path, speaker_name = item
            text_ids_length = text_ids.shape[0]
            
            f = open(feat_path, 'rb')
            mel = np.fromfile(f, dtype='float32')
            mel = np.resize(mel, (-1, config.n_mels))
            mel_length = mel.shape[0]
            f.close()
            speaker = 0
            
            if mel_length < config.mel_length_threshold:
                continue
            
            # create guided attention (default).
            g_attention = _guided_attention(
                text_ids_length,
                mel_length,
                max_ids_length,
                max_mel_length,
                config.guided_attention
            )
            
            yield { "input_ids": text_ids,
                     "input_lengths": text_ids_length,
                     "speaker_ids": speaker,
                     "mel_gts": mel,
                     "mel_lengths": mel_length,
                     "g_attentions": g_attention }

    output_types={ "input_ids": tf.int32,
                        "input_lengths": tf.int32,
                        "speaker_ids": tf.int32,
                        "mel_gts": tf.float32,
                        "mel_lengths": tf.int32, 
                        "g_attentions": tf.float32 }
                                                  
    datasets = tf.data.Dataset.from_generator(_generator, output_types=output_types)
    datasets = datasets.cache()
    datasets = datasets.shuffle(len(items),reshuffle_each_iteration=True)
            
    padding_values = {
        "input_ids": 0,
        "input_lengths": 0,
        "speaker_ids": 0,
        "mel_gts": 0.0,
        "mel_lengths": 0,
        "g_attentions": -1.0
    }
    
    padded_shapes = {
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
    
class Tacotron2Trainer(Seq2SeqBasedTrainer):
    """Tacotron2 Trainer class based on Seq2SeqBasedTrainer."""
    def __init__(self, config, strategy, steps=0, epochs=0, is_mixed_precision=False):
        super(Tacotron2Trainer, self).__init__(
            steps=steps,
            epochs=epochs,
            config=config,
            strategy=strategy,
            is_mixed_precision=is_mixed_precision,
        )
        self.list_metrics_name = [
            "stop_token_loss",
            "mel_loss_before",
            "mel_loss_after",
            "guided_attention_loss"
        ]
        self.init_train_eval_metrics(self.list_metrics_name)
        self.reset_states_train()
        self.reset_states_eval()

        self._config = config

    def compile(self, model, optimizer):
        super().compile(model, optimizer)
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

    def _train_step(self, batch):
        if self._already_apply_input_signature is False:
            self.one_step_forward = tf.function(self._one_step_forward, experimental_relax_shapes=True)
            self.one_step_evaluate = tf.function(self._one_step_evaluate, experimental_relax_shapes=True)
            self.one_step_predict = tf.function(self._one_step_predict, experimental_relax_shapes=True)
            self._already_apply_input_signature = True
            
        # run one_step_forward
        self.one_step_forward(batch)        # error
        
        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def compute_per_example_losses(self, batch, outputs):
        (
            decoder_output,
            post_mel_outputs,
            stop_token_predictions,
            alignment_historys
        ) = outputs

        mel_loss_before = calculate_3d_loss(batch["mel_gts"], decoder_output, loss_fn=self.mae)
        mel_loss_after = calculate_3d_loss(batch["mel_gts"], post_mel_outputs, loss_fn=self.mae)

        # calculate stop_loss
        max_mel_length = tf.reduce_max(batch["mel_lengths"])
        stop_gts = tf.expand_dims(tf.range(tf.reduce_max(max_mel_length), dtype=tf.int32), 0)  # [1, max_len]
        stop_gts = tf.tile(stop_gts, [tf.shape(batch["mel_lengths"])[0], 1])  # [B, max_len]
        stop_gts = tf.cast(tf.math.greater_equal(stop_gts, tf.expand_dims(batch["mel_lengths"], 1)),tf.float32)

        stop_token_loss = calculate_2d_loss(stop_gts, stop_token_predictions, loss_fn=self.binary_crossentropy)
        attention_masks = tf.cast(tf.math.not_equal(batch["g_attentions"], -1.0), tf.float32)
        loss_att = tf.reduce_sum(tf.abs(alignment_historys * batch["g_attentions"]) * attention_masks,axis=[1, 2])
        loss_att /= tf.reduce_sum(attention_masks, axis=[1, 2])

        per_example_losses = (stop_token_loss + mel_loss_before + mel_loss_after + loss_att)

        dict_metrics_losses = {
            "stop_token_loss": stop_token_loss,
            "mel_loss_before": mel_loss_before,
            "mel_loss_after": mel_loss_after,
            "guided_attention_loss": loss_att
        }
        return per_example_losses, dict_metrics_losses

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        import matplotlib.pyplot as plt

        # predict with tf.function for faster.
        outputs = self.one_step_predict(batch)
        (
            decoder_output,
            mel_outputs,
            stop_token_predictions,
            alignment_historys,
        ) = outputs
        mel_gts = batch["mel_gts"]

        # convert to tensor.
        # here we just take a sample at first replica.
        try:
            mels_before = decoder_output.values[0].numpy()
            mels_after = mel_outputs.values[0].numpy()
            mel_gts = mel_gts.values[0].numpy()
            alignment_historys = alignment_historys.values[0].numpy()
        except Exception:
            mels_before = decoder_output.numpy()
            mels_after = mel_outputs.numpy()
            mel_gts = mel_gts.numpy()
            alignment_historys = alignment_historys.numpy()

        # check directory
        dirname = os.path.join(self._config.outdir, f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (mel_gt, mel_before, mel_after, alignment_history) in enumerate(
            zip(mel_gts, mels_before, mels_after, alignment_historys), 1
        ):
            mel_gt = tf.reshape(mel_gt, (-1, self.config.n_mels)).numpy()
            mel_before = tf.reshape(mel_before, (-1, self.config.n_mels)).numpy()
            mel_after = tf.reshape(mel_after, (-1, self.config.n_mels)).numpy()

            # plot figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            im = ax1.imshow(np.rot90(mel_gt), aspect="auto", interpolation="none")
            ax1.set_title("Target Mel-Spectrogram")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax1)
            ax2.set_title(f"Predicted Mel-before-Spectrogram @ {self.steps} steps")
            im = ax2.imshow(np.rot90(mel_before), aspect="auto", interpolation="none")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax2)
            ax3.set_title(f"Predicted Mel-after-Spectrogram @ {self.steps} steps")
            im = ax3.imshow(np.rot90(mel_after), aspect="auto", interpolation="none")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax3)
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # plot alignment
            figname = os.path.join(dirname, f"{idx}_alignment.png")
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.set_title(f"Alignment @ {self.steps} steps")
            im = ax.imshow(
                alignment_history, aspect="auto", origin="lower", interpolation="none"
            )
            fig.colorbar(im, ax=ax)
            xlabel = "Decoder timestep"
            plt.xlabel(xlabel)
            plt.ylabel("Encoder timestep")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()
    
def main():
    """Run training process."""
    parser = argparse.ArgumentParser(description="Train Tacotron2")
    parser.add_argument("--outdir", type=str, required=True, help="directory to save checkpoints.")
    parser.add_argument("--rootdir", type=str, required=True, help="dataset directory root")
    parser.add_argument("--resume",default="",type=str,nargs="?",help='checkpoint file path to resume training. (default="")')
    parser.add_argument("--verbose",type=int,default=1,help="logging level. higher is more logging. (default=1)")
    parser.add_argument("--mixed_precision",default=0,type=int,help="using mixed precision for generator or not.")
    args = parser.parse_args()
    
    # set mixed precision config
    if args.mixed_precision == 1:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    args.mixed_precision = bool(args.mixed_precision)
    
    # set logger
    log_format = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,stream=sys.stdout,format=log_format)
    elif args.verbose > 0:
        logging.basicConfig(level=logging.INFO,stream=sys.stdout,format=log_format)
    else:
        logging.basicConfig(level=logging.WARN,stream=sys.stdout,format=log_format)
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence(checkpoint)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    # select processor
    processor = JSpeechProcessor(args.rootdir)     # for test
    config = Config(args.outdir, processor.vocab_size())
    
    max_mel_length = processor.max_feat_size() // 4 // config.n_mels
    max_ids_length = processor.max_ids_length()
    
    # split train and test 
    train_split, valid_split = train_test_split(processor.items, test_size=config.test_size,random_state=42,shuffle=True)
    train_dataset = generate_datasets(train_split, config, max_mel_length, max_ids_length)
    valid_dataset = generate_datasets(valid_split, config, max_mel_length, max_ids_length)
     
    # define trainer
    trainer = Tacotron2Trainer(
        config=config,
        strategy=STRATEGY,
        steps=0,
        epochs=0,
        is_mixed_precision=args.mixed_precision
    )
    
    with STRATEGY.scope():
        # define model.
        tacotron2 = TFTacotron2(config=config, training=True, name="tacotron2")
        #build
        input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        input_lengths = np.array([9])
        speaker_ids = np.array([0])
        mel_outputs = np.random.normal(size=(1, 50, config.n_mels)).astype(np.float32)
        mel_lengths = np.array([50])
        tacotron2(input_ids,input_lengths,speaker_ids,mel_outputs,mel_lengths,10,training=True)
        tacotron2.summary()

        # AdamW for tacotron2
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=config.initial_learning_rate,
            decay_steps=config.decay_steps,
            end_learning_rate=config.end_learning_rate,
        )

        learning_rate_fn = WarmUp(
            initial_learning_rate=config.initial_learning_rate,
            decay_schedule_fn=learning_rate_fn,
            warmup_steps=int(config.train_max_steps* config.warmup_proportion),
        )

        optimizer = AdamWeightDecay(
            learning_rate=learning_rate_fn,
            weight_decay_rate=config.weight_decay,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
        )

        _ = optimizer.iterations

    # compile trainer
    trainer.compile(model=tacotron2, optimizer=optimizer)

    # start training
    try:
        trainer.fit(
            train_dataset,
            valid_dataset,
            saved_path=os.path.join(args.outdir, "checkpoints/"),
            resume=args.resume
        )
    except KeyboardInterrupt:
        trainer.save_checkpoint()
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
