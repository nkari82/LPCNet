import re
import tensorflow as tf

from tacotron2 import TFTacotron2
 
class Config(object):
    def __init__(
        self,
        vocab_size=10,   #length of symbol
        n_speakers=1):  
        
        # tacotron2 params
        self.vocab_size = vocab_size
        self.embedding_hidden_size = 512
        self.initializer_range = 0.02
        self.n_speakers = n_speakers
        self.layer_norm_eps = 1e-6
        self.embedding_dropout_prob = 0.1
        self.n_conv_encoder = 5
        self.encoder_conv_filters = 512
        self.encoder_conv_kernel_sizes = 5
        self.encoder_conv_activation = 'relu'
        self.encoder_conv_dropout_rate = 0.5
        self.encoder_lstm_units = 256
        self.n_prenet_layers = 2
        self.prenet_units = 256
        self.prenet_activation = 'relu'
        self.prenet_dropout_rate = 0.5
        self.decoder_lstm_units = 1024
        self.n_lstm_decoder = 1
        self.attention_type = 'lsa'
        self.attention_dim = 128
        self.attention_filters = 32
        self.attention_kernel = 31
        self.n_mels = 80            # lpctron is 20mels
        self.reduction_factor = 1
        self.n_conv_postnet = 5
        self.postnet_conv_filters = 512
        self.postnet_conv_kernel_sizes = 5
        self.postnet_dropout_rate = 0.1
                    
def main():
    config = Config()

    # define model.
    model = TFTacotron2(config, training=True, name="tacotron2")
    model._build()
    model.summary()
        
    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False
    )

    nb_epochs = 1
    batch_size = 32
    
    model.compile(optimizer=opt, loss=['mean_absolute_error', 'mean_absolute_error'])
    
    # first argument ['input_lengths', 'speaker_ids', 'mel_gts', 'mel_lengths'].
    model.fit(batch_size=batch_size, epochs=nb_epochs)

if __name__ == "__main__":
    main()
