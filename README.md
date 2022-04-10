# Spectrogram Player
Going back and forth from the time to the frequency domain is an everyday task in audio processing. But can you go back in time without any phase information? The goal of this project is to compare data-driven (e.g., CNN-based) and hand-crafted (e.g., Griffin-Lim algorithm) solutions to reconstruct the audio waveform starting from a spectrogram (i.e., STFT magnitude with no phase information). Evaluation should be performed on both speech and music.

# Review of the methods:
| Methods     | Input      | Output | Link | Parameters
| ----------- | ----------- |------- | ---- | ---- |
| Griffin-Lim | Linear spectrogram | waveform | https://librosa.org/doc/main/generated/librosa.griffinlim.html |
| MelGAN   | Spectrogram   | Waveform | https://github.com/descriptinc/melgan-neurips | https://github.com/descriptinc/melgan-neurips/blob/6488045bfba1975602288de07a58570c7b4d66ea/mel2wav/modules.py#L26
| WaveNet  | Mel-spectrogram | Waveform | https://github.com/auspicious3000/autovc/blob/master/vocoder.ipynb |

# Useful links:
https://github.com/KnurpsBram/AutoVC_WavenetVocoder_GriffinLim_experiments
https://github.com/auspicious3000/autovc
