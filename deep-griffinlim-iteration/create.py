import multiprocessing as mp
import os
from argparse import ArgumentParser

import librosa
import numpy as np
import soundfile as sf
from numpy import ndarray
from tqdm import tqdm

from hparams import hp, test_blacklist, music_blacklist
from pathlib import Path


def save_feature(i_speech: int, s_path_speech: str, speech: ndarray) -> tuple:
    spec_clean = np.ascontiguousarray(librosa.stft(speech, **hp.kwargs_stft))
    mag_clean = np.ascontiguousarray(np.abs(spec_clean)[..., np.newaxis])
    signal_power = np.mean(np.abs(speech)**2)
    list_dict = []
    list_snr_db = []
    for _ in enumerate(args.num_snr):
        snr_db = -6*np.random.rand()
        list_snr_db.append(snr_db)
        snr = librosa.db_to_power(snr_db)
        noise_power = signal_power / snr
        noisy = speech + np.sqrt(noise_power) * np.random.randn(len(speech))
        spec_noisy = librosa.stft(noisy, **hp.kwargs_stft)
        spec_noisy = np.ascontiguousarray(spec_noisy)

        list_dict.append(
            dict(spec_noisy=spec_noisy,
                 spec_clean=spec_clean,
                 mag_clean=mag_clean,
                 path_speech=s_path_speech,
                 length=len(speech),
                 )
        )
    return i_speech, list_snr_db, list_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('kind_data', choices=('TRAIN', 'TEST'))

    args = hp.parse_argument(parser, print_argument=False)
    args.kind_data = args.kind_data.lower()

    # if args.kind_data == 'test':
    blacklist = music_blacklist

    path_feature = hp.dict_path[f'feature_{args.kind_data}']
    os.makedirs(path_feature, exist_ok=True)

    if args.kind_data == 'train':
        path_speech_folder = hp.dict_path[f'speech_{args.kind_data}']
        flist_speech = (list(path_speech_folder.glob('**/*.WAV')) +
                        list(path_speech_folder.glob('**/*.wav')))

        path_music_folder = hp.dict_path[f'music_{args.kind_data}']
        flist_music = (list(path_music_folder.glob('**/*.WAV')) +
                       list(path_music_folder.glob('**/*.wav')))

        loop_music = tqdm(enumerate(flist_music), total=len(flist_music))
        errors = 0

        for i_music, path_music in loop_music:
            filename = str(path_music).split('/')[-1]
            if filename in blacklist:
                continue
            try:
                music = sf.read(str(path_music))[0].astype(np.float32)
            except Exception as e:
                errors += 1
                print(filename + ' skipped.')
                continue
            i_music, list_snr_db, list_dict = save_feature(i_music, str(path_music), music)
            for snr_db, dict_result in zip(list_snr_db, list_dict):
                np.savez(path_feature / hp.form_feature.format(filename, snr_db),
                         **dict_result,
                         )

        loop_speech = tqdm(enumerate(flist_speech), total=len(flist_speech))
        dataset_trim = 5000 - (i_music - len(blacklist))

        for i_speech, path_speech in loop_speech:
            if i_speech == dataset_trim:
                break

            filename = str(path_speech).split('/')[-1]
            # path_speech = os.path.join(path_speech_folder, path_speech)
            speech = sf.read(str(path_speech))[0].astype(np.float32)
            i_speech, list_snr_db, list_dict = save_feature(i_speech, str(path_speech), speech)
            for snr_db, dict_result in zip(list_snr_db, list_dict):
                np.savez(path_feature / hp.form_feature.format(filename, snr_db),
                         **dict_result,
                         )

    else:
        # load test files:
        path_speech = './data/dgl_test_files/signals_speech.npy'
        path_music = './data/dgl_test_files/signals_music.npy'
        signals_speech = np.load(path_speech, allow_pickle=True)
        signals_music = np.load(path_music, allow_pickle=True)

        path_feature_speech = Path(os.path.join(path_feature, 'speech'))
        os.makedirs(path_feature_speech, exist_ok=True)

        loop_speech = tqdm(enumerate(signals_speech), total=len(signals_speech))

        for i_speech, file in loop_speech:
            # path_speech = os.path.join(path_speech_folder, path_speech)
            i_speech, list_snr_db, list_dict = save_feature(i_speech, str(path_speech), file)
            for snr_db, dict_result in zip(list_snr_db, list_dict):
                np.savez(path_feature_speech / hp.form_feature_test.format(i_speech),
                         **dict_result,
                         )

        loop_music = tqdm(enumerate(signals_music), total=len(signals_music))

        path_feature_music = Path(os.path.join(path_feature, 'music'))
        os.makedirs(path_feature_music, exist_ok=True)

        for i_music, file in loop_music:
            i_music, list_snr_db, list_dict = save_feature(i_music, path_music, file)
            for snr_db, dict_result in zip(list_snr_db, list_dict):
                np.savez(path_feature_music / hp.form_feature_test.format(i_music),
                         **dict_result,
                         )
