# MellotronCPU
Mellotron singing synthesizer using CPU

## Insallation

Download pretrained model checkpoints from nvidia/mellotron repository and specify the paths [here](https://github.com/mathigatti/MellotronCPU/blob/master/run_mellotron.py#L21)


## Usage

Check `Playground.ipyng`


### About musicXML Format

1. The characters must be in [a-zA-Z]
2. Each word must start with an upper case
3. Every word must exist in the cmu_dictionary dictionary. https://en.wikipedia.org/wiki/ARPABET



### Relevant notes 1

In reference to the GST part of mellotron, there is no 1:1 lock. You can use GST the same way as in other repos.

If you want to do inference with the mellotron model however, we additionally extract two things from a reference audio: the rhythm and the pitch which creates the 1:1 correspondence. It's the rhythm that creates the 1:1 correspondence actually. But your automatically-extracted pitch might not make sense if you do not additionally condition on the rhythm.

If you don't want rhythm (which you can disable by using model.inferece()) and pitch conditioning (which you can disable by sending zeros as the pitch), you get essentially tacotron 2 with GST and speaker ids.


### Relevant notes 2

The paper states that "the target speaker, St, would always be found in the training set, while the source text, pitch and rhythm (Ts, Ps, Rs) could be from outside the training set." so I presume there is no need for speaker ids for source audios - it doesn't make sense after all for some arbitrary input audio outside the training set to have a valid speaker id. However in the examples_filelist.txt there is a column for speaker ids. What is the significance of this column?

The model expects a speaker id, so we give it a random speaker id.

https://github.com/NVIDIA/mellotron/issues/18