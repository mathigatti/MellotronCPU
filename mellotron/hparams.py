import tensorflow as tf
from text.symbols import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    class HParams():
        def __init__(self):
            ################################
            # Experiment Parameters        #
            ################################
            self.epochs=50000
            self.iters_per_checkpoint=500
            self.seed=1234
            self.dynamic_loss_scaling=True
            self.fp16_run=False
            self.distributed_run=False
            self.dist_backend="nccl"
            self.dist_url="tcp://localhost:54321"
            self.cudnn_enabled=True
            self.cudnn_benchmark=False
            self.ignore_layers=['speaker_embedding.weight']

            ################################
            # Data Parameters             #
            ################################
            self.training_files='filelists/ljs_audiopaths_text_sid_train_filelist.txt'
            self.validation_files='filelists/ljs_audiopaths_text_sid_val_filelist.txt'
            self.text_cleaners=['english_cleaners']
            self.p_arpabet=1.0
            self.cmudict_path="mellotron/data/cmu_dictionary"

            ################################
            # Audio Parameters             #
            ################################
            self.max_wav_value=32768.0
            self.sampling_rate=22050
            self.filter_length=1024
            self.hop_length=256
            self.win_length=1024
            self.n_mel_channels=80
            self.mel_fmin=0.0
            self.mel_fmax=8000.0
            self.f0_min=80
            self.f0_max=880
            self.harm_thresh=0.25

            ################################
            # Model Parameters             #
            ################################
            self.n_symbols=len(symbols)
            self.symbols_embedding_dim=512

            # Encoder parameters
            self.encoder_kernel_size=5
            self.encoder_n_convolutions=3
            self.encoder_embedding_dim=512

            # Decoder parameters
            self.n_frames_per_step=1 # currently only 1 is supported
            self.decoder_rnn_dim=1024
            self.prenet_dim=256
            self.prenet_f0_n_layers=1
            self.prenet_f0_dim=1
            self.prenet_f0_kernel_size=1
            self.prenet_rms_dim=0
            self.prenet_rms_kernel_size=1
            self.max_decoder_steps=1000
            self.gate_threshold=0.5
            self.p_attention_dropout=0.1
            self.p_decoder_dropout=0.1
            self.p_teacher_forcing=1.0

            # Attention parameters
            self.attention_rnn_dim=1024
            self.attention_dim=128

            # Location Layer parameters
            self.attention_location_n_filters=32
            self.attention_location_kernel_size=31

            # Mel-post processing network parameters
            self.postnet_embedding_dim=512
            self.postnet_kernel_size=5
            self.postnet_n_convolutions=5

            # Speaker embedding
            #self.n_speakers=123
            self.n_speakers=1
            self.speaker_embedding_dim=128

            # Reference encoder
            self.with_gst=True
            self.ref_enc_filters=[32, 32, 64, 64, 128, 128]
            self.ref_enc_size=[3, 3]
            self.ref_enc_strides=[2, 2]
            self.ref_enc_pad=[1, 1]
            self.ref_enc_gru_size=128

            # Style Token Layer
            self.token_embedding_size=256
            self.token_num=10
            self.num_heads=8

            ################################
            # Optimization Hyperparameters #
            ################################
            self.use_saved_learning_rate=False
            self.learning_rate=1e-3
            self.learning_rate_min=1e-5
            self.learning_rate_anneal=50000
            self.weight_decay=1e-6
            self.grad_clip_thresh=1.0
            self.batch_size=32
            self.mask_padding=True  # set model's padded outputs to padded values

    hparams = HParams()

    return hparams
