import os
import librosa
import numpy as np
import soundfile as sf
from numpy import ndarray
from tqdm import tqdm
from pathlib import Path
from hparams import _HyperParameters as hp
from sklearn.metrics import mean_squared_error


def create_for_colab():
    type = hp.test_set
    path_folder = Path(f'{hp.path_colab}/wavs/{type}/{hp.training_type}')
    os.makedirs(path_folder, exist_ok=True)
    repeat = hp.repeat_test
    flist = (list(path_folder.glob(f'**/*.WAV')) +
                    list(path_folder.glob(f'**/*.wav')))

    flist = sorted(flist)

    loop = tqdm(enumerate(flist), total=len(flist))
    results = []

    for i_speech, path_speech in loop:
        file = sf.read(str(path_speech))[0].astype(np.float32)
        results.append(file)

    np.save(os.path.join(hp.path_colab, f'deepgl_{hp.training_type}_{repeat}_{type}.npy'), results)


def order_files():
    type = 'urban' # speech, music, urban

    hp.init_dependent_vars(hp)

    signal_reference = np.load(f'./data/dgl_test_files/signals_{type}.npy', allow_pickle=True)
    to_process_1 = np.load(f'{hp.path_colab}/deepgl_unbiased_{hp.repeat_test}_{type}.npy', allow_pickle=True)
    to_process_2 = np.load(f'{hp.path_colab}/deepgl_biased_{hp.repeat_test}_{type}.npy', allow_pickle=True)

    support_1 = np.zeros(len(signal_reference), dtype=int)
    mse_1 = np.zeros(len(signal_reference), dtype=np.float32)
    support_2 = np.zeros(len(signal_reference), dtype=int)
    mse_2 = np.zeros(len(signal_reference), dtype=np.float32)

    for index, signal_1 in enumerate(signal_reference):
        stft_1 = abs(librosa.stft(signal_1[:16000 * 2], **hp.kwargs_stft))

        for i, signal_2 in enumerate(to_process_1):
            stft_2 = abs(librosa.stft(signal_2[:16000 * 2], **hp.kwargs_stft))
            min_y = min(stft_2.shape[1], stft_1.shape[1])
            stft_2 = stft_2[:, :min_y]
            stft_1_mod = stft_1[:, :min_y]
            print(stft_2.shape, stft_1_mod.shape)
            mse_1[i] = mean_squared_error(stft_2, stft_1_mod)

        min_index = np.argmin(mse_1)
        support_1[index] = min_index

        for i, signal_2 in enumerate(to_process_2):
            stft_2 = abs(librosa.stft(signal_2[:16000 * 2], **hp.kwargs_stft))
            min_y = min(stft_2.shape[1], stft_1.shape[1])
            stft_2 = stft_2[:, :min_y]
            stft_1_mod = stft_1[:, :min_y]
            mse_2[i] = mean_squared_error(stft_2, stft_1_mod)

        min_index = np.argmin(mse_2)
        support_2[index] = min_index

    result_1 = to_process_1[support_1]
    result_2 = to_process_2[support_2]
    file_1 = f'deepgl_unbiased_{hp.repeat_test}_{type}.npy'
    file_2 = f'deepgl_biased_{hp.repeat_test}_{type}.npy'
    np.save(os.path.join(hp.path_colab, file_1), result_1)
    np.save(os.path.join(hp.path_colab, file_2), result_2)

    print(f'Created:\n{file_1}\n{file_2}\n')

def main():
    order_files()


if __name__ == "__main__":
    main()

