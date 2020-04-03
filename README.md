# MellotronCPU
Mellotron singing synthesizer using CPU

## Insallation

- Download pretrained model checkpoints from nvidia/mellotron repository
- Install requirements


## Usage


```
from run_mellotron import *
model = init_model()

filename = "musicXML/last_voice_processed_4.xml"
speaker_id = 0 # Choose some speaker
synthesize(filename, model, bpm=80, speaker_id=speaker_id, outname="sample.wav")
```
