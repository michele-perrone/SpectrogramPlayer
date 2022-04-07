# Spectrogram Player
Going back and forth from the time to the frequency domain is an everyday task in audio processing. But can you go back in time without any phase information? The goal of this project is to compare data-driven (e.g., CNN-based) and hand-crafted (e.g., Griffin-Lim algorithm) solutions to reconstruct the audio waveform starting from a spectrogram (i.e., STFT magnitude with no phase information). Evaluation should be performed on both speech and music.

# Review of the methods:
| Methods     | Input      | Output | Link |
| ----------- | ----------- |------- | ---- |
| Griffin-Lim | Arbitrary spectrogram | waveform | https://librosa.org/doc/main/generated/librosa.griffinlim.html
| MelGAN   | Spectrogram   | Waveform | https://github.com/descriptinc/melgan-neurips

# Useful links:
https://github.com/KnurpsBram/AutoVC_WavenetVocoder_GriffinLim_experiments
https://github.com/auspicious3000/autovc
