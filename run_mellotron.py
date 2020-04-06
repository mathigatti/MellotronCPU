import sys
sys.path.append('mellotron')
sys.path.append('mellotron/waveglow/')

import os
import numpy as np
from scipy.io.wavfile import write
import torch
from hparams import create_hparams
from waveglow.denoiser import Denoiser
from train_utils import load_model
from data_utils import TextMelLoader, TextMelCollate
from text import cmudict, text_to_sequence
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



def synthesize1(filename, model, bpm=80, speaker_id=1, outname="sample.wav"):
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
		audio = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.66), 0.01)[0, 0]
		audio = audio.cpu().numpy()
		pan = 0
		audio = panner(audio, pan)
		audio_stereo[:audio.shape[0]] += audio
		write(outname, sampling_rate, audio)

import librosa
from layers import TacotronSTFT

def load_mel(path):
	hparams = create_hparams()
	stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
	                hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
	                hparams.mel_fmax)
	audio, sampling_rate = librosa.core.load(path, sr=hparams.sampling_rate)
	audio = torch.from_numpy(audio)
	if sampling_rate != hparams.sampling_rate:
	    raise ValueError("{} SR doesn't match target {} SR".format(
	        sampling_rate, stft.sampling_rate))
	audio_norm = audio / hparams.max_wav_value
	audio_norm = audio_norm.unsqueeze(0)
	audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
	melspec = stft.mel_spectrogram(audio_norm)
	melspec = melspec.cpu()
	return melspec

def synthesize2(model, audio_path, text, source_speaker_id, target_speaker_id=1, outname="sample.wav"):
	tacotron, waveglow, denoiser = model
	with open('temp.txt','w') as f:
		f.write(f"{audio_path}|{text}|{source_speaker_id}")
	arpabet_dict = cmudict.CMUDict('mellotron/data/cmu_dictionary')
	hparams = create_hparams()
	dataloader = TextMelLoader("temp.txt", hparams)
	datacollate = TextMelCollate(1)

	file_idx = 0
	audio_path, text, sid = dataloader.audiopaths_and_text[file_idx]

	# get audio path, encoded text, pitch contour and mel for gst
	text_encoded = torch.LongTensor(text_to_sequence(text, hparams.text_cleaners, arpabet_dict))[None, :].cpu()
	pitch_contour = dataloader[file_idx][3][None].cpu()
	mel = load_mel(audio_path)
	print(audio_path, text)

	# load source data to obtain rhythm using tacotron 2 as a forced aligner
	x, y = tacotron.parse_batch(datacollate([dataloader[file_idx]]))

	# For changing the pitch
	pitch_contour2 = pitch_contour.data.cpu().numpy().copy()
	#pitch_contour2[pitch_contour2 > 0] -= 45.
	#pitch_contour2[pitch_contour2 > 0] = 150.
	pitch_contour2 = torch.Tensor(pitch_contour2).cpu()

	with torch.no_grad():
	    # get rhythm (alignment map) using tacotron 2
	    mel_outputs, mel_outputs_postnet, gate_outputs, rhythm = tacotron.forward(x)
	    rhythm = rhythm.permute(1, 0, 2)

	speaker_id = torch.LongTensor([target_speaker_id]).cpu()

	sampling_rate = 22050

	with torch.no_grad():
		mel_outputs, mel_outputs_postnet, gate_outputs, _ = tacotron.inference_noattention((text_encoded, mel, speaker_id, pitch_contour2, rhythm))
		audio = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.66), 0.03)[0, 0]
		audio = audio.cpu().numpy()
		pan = 0
		audio = panner(audio, pan)
		write(outname, sampling_rate, audio)
	os.remove("temp.txt")