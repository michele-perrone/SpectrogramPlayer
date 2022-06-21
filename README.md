# Spectrogram Player
Going back and forth from the time to the frequency domain is an everyday task in audio processing. But can you go back in time without any phase information? The goal of this project is to compare data-driven (e.g., CNN-based) and hand-crafted (e.g., Griffin-Lim algorithm) solutions to reconstruct the audio waveform starting from a spectrogram (i.e., STFT magnitude with no phase information). Evaluation should be performed on both speech and music.

# Review of the methods:
| Methods     | Input      | Output | Link | Parameters
| ----------- | ----------- |------- | ---- | ---- |
| Griffin-Lim | Linear spectrogram | waveform | https://librosa.org/doc/main/generated/librosa.griffinlim.html |
| MelGAN   | Mel-Spectrogram   | Waveform | https://github.com/descriptinc/melgan-neurips | https://github.com/descriptinc/melgan-neurips/blob/6488045bfba1975602288de07a58570c7b4d66ea/mel2wav/modules.py#L26
| HiFiGan  | Mel-spectrogram | Waveform | https://github.com/NVIDIA/NeMo/blob/75c166864541f8b90d525868512f0e4d8dac15da/nemo/collections/tts/models/hifigan.py | 
| SqueezeWave | Mel-spectrogram | Waveform | https://github.com/NVIDIA/NeMo/blob/75c166864541f8b90d525868512f0e4d8dac15da/nemo/collections/tts/models/squeezewave.py |
| UniGlow | Mel-spectrogram | Waveform | https://github.com/NVIDIA/NeMo/blob/75c166864541f8b90d525868512f0e4d8dac15da/nemo/collections/tts/models/uniglow.py |
| UnivNet | Mel-spectrogram | Waveform| https://github.com/mindslab-ai/univnet | 
| Deep Griffin-Lim | Mel-spectrogram | Waveform | https://github.com/Sytronik/deep-griffinlim-iteration | 

# Deep Griffin-Lim iteration:
Here we describe the steps to train the DGL model. Minor changes were made, w.r.t. the original Repo https://github.com/Sytronik/deep-griffinlim-iteration, to accomodate the use of multiple dataset and update legacy libraries.

## Folder structure
- **model**: contains the DGL model
- **create.py**: where to preprocess the dataset to be ready for the training. To process the files enter: python create.py TRAIN/TEST --num_snr YOUR_CHOICE. We tested our model with num_snr=3.
- **create_result_file.py**: here the individual wav results are loaded and saved in a single numpy array. This way, we can utilize them directly in our main Jupiter Notebook.
- **dataset.py**: used by other modules, subclass of the Pytorch Dataset class.
- **hparams.py**: where paths are set as well as training/testing conditions and other general parameters.
- **main.py**: here the training/testing conditions are set. To train or test use: python main.py --train/test
- **tbwriter.py**: this module writes the logs of the training, as well as the result wav files.
- **train.py**: contains the code for training and testing.
- **utils.py**: various methods used by the modules.
