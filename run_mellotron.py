import sys
sys.path.append('mellotron')
sys.path.append('mellotron/waveglow/')

import numpy as np
from scipy.io.wavfile import write
import torch
from hparams import create_hparams
from waveglow.denoiser import Denoiser
from train_utils import load_model
from data_utils import TextMelLoader, TextMelCollate
from text import cmudict
from mellotron_utils import get_data_from_musicxml

def panner(signal, angle):
    angle = np.radians(angle)
    left = np.sqrt(2)/2.0 * (np.cos(angle) - np.sin(angle)) * signal
    right = np.sqrt(2)/2.0 * (np.cos(angle) + np.sin(angle)) * signal
    return np.dstack((left, right))[0]

def init_model():
	hparams = create_hparams()

	checkpoint_path = "checkpoints/mellotron_libritts.pt"
	tacotron = load_model(hparams).cpu().eval()
	tacotron.load_state_dict(torch.load(checkpoint_path,map_location=torch.device('cpu'))['state_dict'])

	waveglow_path = 'checkpoints/waveglow_256channels_v4.pt'
	waveglow = torch.load(waveglow_path,map_location=torch.device('cpu'))['model'].cpu().eval()
	denoiser = Denoiser(waveglow).cpu().eval()
	return (tacotron, waveglow, denoiser)

def synthesize(filename, model, bpm=80, speaker_id=1, outname="sample.wav"):
	tacotron, waveglow, denoiser = model
	data = get_data_from_musicxml(filename, bpm)

	sampling_rate = 22050
	frequency_scaling = 0.4
	n_seconds = 90
	audio_stereo = np.zeros((sampling_rate*n_seconds, 2), dtype=np.float32)

	data_v = list(data.values())[0]

	rhythm = data_v['rhythm'].cpu()
	pitch_contour = data_v['pitch_contour'].cpu()
	text_encoded = data_v['text_encoded'].cpu()

	speaker_id = torch.LongTensor([speaker_id]).cpu()

	with torch.no_grad():
		some_number_i_dont_know_what_is_this = 0 # Seems to be a number from 0 to 10
		mel_outputs, mel_outputs_postnet, gate_outputs, alignments_transfer = tacotron.inference_noattention((text_encoded, some_number_i_dont_know_what_is_this, speaker_id, pitch_contour*frequency_scaling, rhythm))
		audio = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[0, 0]
		audio = audio.cpu().numpy()
		pan = 0
		audio = panner(audio, pan)
		audio_stereo[:audio.shape[0]] += audio
		write(outname, sampling_rate, audio)