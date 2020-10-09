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
"""Train FastSpeech2."""
import tensorflow as tf
import sys
import argparse
import logging
import os
import collections
import numpy as np
import tensorflow_tts as tts
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow_tts.models import TFFastSpeech2
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

    SelfAttentionParams = collections.namedtuple(
        "SelfAttentionParams",
        [
            "n_speakers",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "attention_head_size",
            "intermediate_size",
            "intermediate_kernel_size",
            "hidden_act",
            "output_attentions",
            "output_hidden_states",
            "initializer_range",
            "hidden_dropout_prob",
            "attention_probs_dropout_prob",
            "layer_norm_eps",
            "max_position_embeddings",
        ],
    )

    def __init__(self,outdir,batch_size=32,vocab_size=150,n_speakers=1):
        # fastspeech2 params
        self.vocab_size = vocab_size
        self.n_speakers = n_speakers
        self.encoder_hidden_size = 256
        self.encoder_num_hidden_layers = 3
        self.encoder_num_attention_heads = 2
        self.encoder_attention_head_size = 16  # in v1, = 384//2
        self.encoder_intermediate_size = 1024
        self.encoder_intermediate_kernel_size = 3
        self.encoder_hidden_act = "mish"
        self.decoder_hidden_size = 256
        self.decoder_num_hidden_layers = 3
        self.decoder_num_attention_heads = 2
        self.decoder_attention_head_size = 16  # in v1, = 384//2
        self.decoder_intermediate_size = 1024
        self.decoder_intermediate_kernel_size = 3
        self.decoder_hidden_act = "mish"
        self.variant_prediction_num_conv_layers = 2
        self.variant_predictor_filter = 256
        self.variant_predictor_kernel_size = 3
        self.variant_predictor_dropout_rate = 0.5
        self.num_mels = 20
        self.hidden_dropout_prob = 0.2
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 2048
        self.initializer_range = 0.02
        self.layer_norm_eps = 1e-5
        self.output_attentions = False
        self.output_hidden_states = False
    
        self.duration_predictor_dropout_probs = 0.1
        self.num_duration_conv_layers = 2
        self.duration_predictor_filters = 256
        self.duration_predictor_kernel_sizes = 3

        # postnet
        self.n_conv_postnet = 5
        self.postnet_conv_filters = 512
        self.postnet_conv_kernel_sizes = 5
        self.postnet_dropout_rate = 0.1
        
        # encoder params
        self.encoder_self_attention_params = self.SelfAttentionParams(
            n_speakers=self.n_speakers,
            hidden_size=self.encoder_hidden_size,
            num_hidden_layers=self.encoder_num_hidden_layers,
            num_attention_heads=self.encoder_num_attention_heads,
            attention_head_size=self.encoder_attention_head_size,
            hidden_act=self.encoder_hidden_act,
            intermediate_size=self.encoder_intermediate_size,
            intermediate_kernel_size=self.encoder_intermediate_kernel_size,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
            initializer_range=self.initializer_range,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            layer_norm_eps=self.layer_norm_eps,
            max_position_embeddings=self.max_position_embeddings,
        )

        # decoder params
        self.decoder_self_attention_params = self.SelfAttentionParams(
            n_speakers=self.n_speakers,
            hidden_size=self.decoder_hidden_size,
            num_hidden_layers=self.decoder_num_hidden_layers,
            num_attention_heads=self.decoder_num_attention_heads,
            attention_head_size=self.decoder_attention_head_size,
            hidden_act=self.decoder_hidden_act,
            intermediate_size=self.decoder_intermediate_size,
            intermediate_kernel_size=self.decoder_intermediate_kernel_size,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
            initializer_range=self.initializer_range,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            layer_norm_eps=self.layer_norm_eps,
            max_position_embeddings=self.max_position_embeddings,
        )
                
        # data
        self.batch_size = batch_size
        self.test_size = 0.05
        self.mel_length_threshold = 0
        self.guided_attention = 0.2         # unused
        
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
        
def generate_datasets(items, config, f0_stat, energy_stat):

    def _average_by_duration(x, durs):
        mel_len = durs.sum()
        durs_cum = np.cumsum(np.pad(durs, (1, 0)))

        # calculate charactor f0/energy
        x_char = np.zeros((durs.shape[0],), dtype=np.float32)
        for idx, start, end in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
            values = x[start:end][np.where(x[start:end] != 0.0)[0]]
            x_char[idx] = np.mean(values) if len(values) > 0 else 0.0  # np.mean([]) = nan.

        return x_char.astype(np.float32)
    
    @tf.function(input_signature=[tf.TensorSpec(None, tf.float32), tf.TensorSpec(None, tf.int32)])
    def _tf_average_by_duration(x, durs):
        return tf.numpy_function(_average_by_duration, [x, durs], tf.float32)
            
    def _norm_mean_std(x, mean, std):
        zero_idxs = np.where(x == 0.0)[0]
        x = (x - mean) / std
        x[zero_idxs] = 0.0
        return x
        
    def _generator():
        for item in items:
            _, text_seq, feat_path, f0_path, energy_path, duration_path, _ = item
            
            with open(feat_path, 'rb') as f:
                mel = np.fromfile(f, dtype='float32')
                mel = np.resize(mel, (-1, config.num_mels))

            mel_length = mel.shape[0]

            if f is None or mel_length < config.mel_length_threshold:
                continue

            with open(f0_path, 'rb') as f:
                f0 = np.fromfile(f, dtype='float32')
            
            with open(energy_path, 'rb') as f:
                energy = np.fromfile(f, dtype='float32')
                
            with open(duration_path, 'rb') as f:
                duration = np.fromfile(f, dtype='int32')
            
            f0 = _norm_mean_std(f0, f0_stat[0], f0_stat[1])
            energy = _norm_mean_std(energy, energy_stat[0], energy_stat[1])
            
            # calculate charactor f0/energy
            f0 = _tf_average_by_duration(f0, duration)
            energy = _tf_average_by_duration(energy, duration)
            
            data = {
                "input_ids": text_seq,
                "speaker_ids": 0,
                "duration_gts": duration,
                "f0_gts": f0,
                "energy_gts": energy,
                "mel_gts": mel,
                "mel_lengths": mel_length
            }
            
            yield data;

    output_types = { 
            "input_ids": tf.int32,
            "speaker_ids": tf.int32,
            "duration_gts": tf.int32,
            "f0_gts": tf.float32,
            "energy_gts": tf.float32,
            "mel_gts": tf.float32,
            "mel_lengths": tf.int32
        }
                                                  
    datasets = tf.data.Dataset.from_generator(_generator, output_types=output_types)
    datasets = datasets.cache()
    datasets = datasets.shuffle(len(items),reshuffle_each_iteration=True)
            
    padded_shapes = {
        "input_ids": [None],
        "speaker_ids": [],
        "duration_gts": [None],
        "f0_gts": [None],
        "energy_gts": [None],
        "mel_gts": [None, config.num_mels],      
        "mel_lengths": []
    }
    
    datasets = datasets.padded_batch(config.batch_size * STRATEGY.num_replicas_in_sync, padded_shapes=padded_shapes)
    datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
    
    return datasets
    
class FastSpeech2Trainer(Seq2SeqBasedTrainer):
    """FastSpeech2 Trainer class based on FastSpeechTrainer."""
    def __init__(self, config, strategy, steps=0, epochs=0, is_mixed_precision=False):
        super(FastSpeech2Trainer, self).__init__(
            steps=steps,
            epochs=epochs,
            config=config,
            strategy=strategy,
            is_mixed_precision=is_mixed_precision,
        )
        self.list_metrics_name = [
            "duration_loss",
            "f0_loss",
            "energy_loss",
            "mel_loss_before",
            "mel_loss_after",
        ]
        self.init_train_eval_metrics(self.list_metrics_name)
        self.reset_states_train()
        self.reset_states_eval()

    def compile(self, model, optimizer):
        super().compile(model, optimizer)
        self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

    def compute_per_example_losses(self, batch, outputs):
        mel_before, mel_after, duration_outputs, f0_outputs, energy_outputs = outputs

        log_duration = tf.math.log(tf.cast(tf.math.add(batch["duration_gts"], 1), tf.float32))
        duration_loss = calculate_2d_loss(log_duration, duration_outputs, self.mse)
        f0_loss = calculate_2d_loss(batch["f0_gts"], f0_outputs, self.mse)
        energy_loss = calculate_2d_loss(batch["energy_gts"], energy_outputs, self.mse)
        mel_loss_before = calculate_3d_loss(batch["mel_gts"], mel_before, self.mae)
        mel_loss_after = calculate_3d_loss(batch["mel_gts"], mel_after, self.mae)

        per_example_losses = (duration_loss + f0_loss + energy_loss + mel_loss_before + mel_loss_after)

        dict_metrics_losses = {
            "duration_loss": duration_loss,
            "f0_loss": f0_loss,
            "energy_loss": energy_loss,
            "mel_loss_before": mel_loss_before,
            "mel_loss_after": mel_loss_after,
        }

        return per_example_losses, dict_metrics_losses

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        import matplotlib.pyplot as plt

        # predict with tf.function.
        outputs = self.one_step_predict(batch)

        mels_before, mels_after, *_ = outputs
        mel_gts = batch["mel_gts"]

        # convert to tensor.
        # here we just take a sample at first replica.
        try:
            mels_before = mels_before.values[0].numpy()
            mels_after = mels_after.values[0].numpy()
            mel_gts = mel_gts.values[0].numpy()
        except Exception:
            mels_before = mels_before.numpy()
            mels_after = mels_after.numpy()
            mel_gts = mel_gts.numpy()

        # check directory
        dirname = os.path.join(self.config.outdir, f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (mel_gt, mel_before, mel_after) in enumerate(zip(mel_gts, mels_before, mels_after), 0):
            mel_gt = tf.reshape(mel_gt, (-1, self.config.num_mels)).numpy()
            mel_before = tf.reshape(mel_before, (-1, self.config.num_mels)).numpy()
            mel_after = tf.reshape(mel_after, (-1, self.config.num_mels)).numpy()

            # plit figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            im = ax1.imshow(np.rot90(mel_gt), aspect="auto", interpolation="none")
            ax1.set_title("Target Mel-Spectrogram")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax1)
            ax2.set_title("Predicted Mel-before-Spectrogram")
            im = ax2.imshow(np.rot90(mel_before), aspect="auto", interpolation="none")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax2)
            ax3.set_title("Predicted Mel-after-Spectrogram")
            im = ax3.imshow(np.rot90(mel_after), aspect="auto", interpolation="none")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax3)
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

#python train_fastspeech2.py --outdir ./fit_fastspeech2 --rootdir ./datasets/jsut/basic --batch-size 1 --resume ./fit_fastspeech2/checkpoints
#python train_fastspeech2.py --outdir ./fit2_fastspeech2 --rootdir ./datasets/jsut/basic --batch-size 1 --resume ./fit_fastspeech2/checkpoints
def main():
    """Run training process."""
    parser = argparse.ArgumentParser(description="Train Tacotron2")
    parser.add_argument("--outdir", type=str, required=True, help="directory to save checkpoints.")
    parser.add_argument("--rootdir", type=str, required=True, help="dataset directory root")
    parser.add_argument("--resume",default="",type=str,nargs="?",help='checkpoint file path to resume training. (default="")')
    parser.add_argument("--verbose",type=int,default=1,help="logging level. higher is more logging. (default=1)")
    parser.add_argument("--batch-size", default=8, type=int, help="batch size.")
    parser.add_argument("--mixed_precision",default=0,type=int,help="using mixed precision for generator or not.")
    parser.add_argument("--pretrained",default="",type=str,nargs="?",help='pretrained weights .h5 file to load weights from. Auto-skips non-matching layers',)
    args = parser.parse_args()
    
    if args.resume is not None and os.path.isdir(args.resume):
        args.resume = tf.train.latest_checkpoint(args.resume)
    
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
    Processor = JSpeechProcessor   # for test
    
    class Generator(Processor.Generator):
        def __init__(self):
            super().__init__()
            self._scaler_energy = StandardScaler(copy=False)
            self._scaler_f0 = StandardScaler(copy=False)
            self._energy_stat = np.stack((0,0))
            self._f0_stat = np.stack((0,0))
            
        def __call__(self, rootdir, tid, seq, speaker):
            tid, seq, feat_path, speaker = super().__call__(rootdir, tid, seq, speaker)
            
            f0_path = os.path.join(rootdir, "f0", f"{tid}.f0")
            energy_path = os.path.join(rootdir, "energies", f"{tid}.e")
            duration_path = os.path.join(rootdir, "durations", f"{tid}.dur")
            
            with open(f0_path) as f:
                f0 = np.fromfile(f, dtype='float32')
                self._scaler_f0.partial_fit(f0[f0 != 0].reshape(-1, 1))
            
            with open(energy_path) as f:
                energy = np.fromfile(f, dtype='float32')
                self._scaler_energy.partial_fit(energy[energy != 0].reshape(-1, 1))
            
            return tid, seq, feat_path, f0_path, energy_path, duration_path, speaker
         
        def complete(self):
            self._f0_stat = np.stack((self._scaler_f0.mean_, self._scaler_f0.scale_))
            self._energy_stat = np.stack((self._scaler_energy.mean_, self._scaler_energy.scale_))
            
            print("energy stat: mean {}, scale {}".format(self._energy_stat[0], self._energy_stat[1]))
            print("f0 stat: mean {}, scale {}".format(self._f0_stat[0], self._f0_stat[1]))
            
        def energy_stat(self):
            return self._energy_stat
            
        def f0_stat(self):
            return self._f0_stat
    
    generator = Generator()
    processor = Processor(rootdir=args.rootdir, generator=generator)     
    
    config = Config(args.outdir, args.batch_size, processor.vocab_size())
    
    # split train and test 
    train_split, valid_split = train_test_split(processor.items, test_size=config.test_size,random_state=42,shuffle=True)
    train_dataset = generate_datasets(train_split, config, generator.f0_stat(), generator.energy_stat())
    valid_dataset = generate_datasets(valid_split, config, generator.f0_stat(), generator.energy_stat())
    
    # define trainer
    trainer = FastSpeech2Trainer(
        config=config,
        strategy=STRATEGY,
        steps=0,
        epochs=0,
        is_mixed_precision=args.mixed_precision
    )
    
    with STRATEGY.scope():
        # define model
        fastspeech = TFFastSpeech2(config=config)
        
        # build
        fastspeech._build()
        fastspeech.summary()
        
        if len(args.pretrained) > 1:
            fastspeech.load_weights(args.pretrained, by_name=True, skip_mismatch=True)
            logging.info(f"Successfully loaded pretrained weight from {args.pretrained}.")

        # AdamW for fastspeech
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=config.initial_learning_rate,
            decay_steps=config.decay_steps,
            end_learning_rate=config.end_learning_rate,
        )

        learning_rate_fn = WarmUp(
            initial_learning_rate=config.initial_learning_rate,
            decay_schedule_fn=learning_rate_fn,
            warmup_steps=int(config.train_max_steps * config.warmup_proportion)
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
    trainer.compile(model=fastspeech, optimizer=optimizer)

    # start training
    try:
        trainer.fit(train_dataset,valid_dataset,saved_path=os.path.join(config.outdir, "checkpoints/"),resume=args.resume)
    except KeyboardInterrupt:
        trainer.save_checkpoint()
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")
        
if __name__ == "__main__":
    main()

    