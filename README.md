# Spectrogram Player
Going back and forth from the time to the frequency domain is an everyday task in audio processing. But can you go back in time without any phase information? The goal of this project is to compare data-driven (e.g., CNN-based) and hand-crafted (e.g., Griffin-Lim algorithm) solutions to reconstruct the audio waveform starting from a spectrogram (i.e., STFT magnitude with no phase information). Evaluation should be performed on both speech and music.

# Useful links 
MelGan in Pytorch: https://github.com/jaywalnut310/MelGAN-Pytorch/blob/9eb3598e93ac0c5bb80e0b2bb25839b2fa8e19ea/preprocessing.py#L10
