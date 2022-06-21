import contextlib
import os
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Union

import librosa
import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from hparams import hp


def reconstruct_wave(*args: ndarray, n_iter=0, n_sample=-1) -> ndarray:
    """ reconstruct time-domain wave from spectrogram

    :param args: can be (mag_spectrogram, phase_spectrogram) or (complex_spectrogram,)
    :param n_iter: no. of iteration of griffin-lim. 0 for not using griffin-lim.
    :param n_sample: number of samples of output wave
    :return:
    """

    if len(args) == 1:
        spec = args[0].squeeze()
        mag = None
        phase = None
        assert np.iscomplexobj(spec)
    elif len(args) == 2:
        spec = None
        mag = args[0].squeeze()
        phase = args[1].squeeze()
        assert np.isrealobj(mag) and np.isrealobj(phase)
    else:
        raise ValueError

    for _ in range(n_iter - 1):
        if mag is None:
            mag = np.abs(spec)
            phase = np.angle(spec)
            spec = None
        wave = librosa.istft(mag * np.exp(1j * phase), **hp.kwargs_istft)

        phase = np.angle(librosa.stft(wave, **hp.kwargs_stft))

    kwarg_len = dict(length=n_sample) if n_sample != -1 else dict()
    if spec is None:
        spec = mag * np.exp(1j * phase)
    wave = librosa.istft(spec, **hp.kwargs_istft, **kwarg_len)

    return wave


def draw_spectrogram(data: ndarray, to_db=True, show=False, dpi=150, **kwargs):
    """
    
    :param data:
    :param to_db:
    :param show:
    :param dpi:
    :param kwargs: vmin, vmax
    :return: 
    """

    if to_db:
        # data[data == 0] = data[data > 0].min()
        data = librosa.amplitude_to_db(data)
    data = data.squeeze()

    fig, ax = plt.subplots(dpi=dpi,)
    ax.imshow(data,
              cmap=plt.get_cmap('CMRmap'),
              extent=(0, data.shape[1], 0, hp.fs // 2),
              origin='lower', aspect='auto', **kwargs)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(ax.images[0])
    if show:
        fig.show()

    return fig


# noinspection PyAttributeOutsideInit
class AverageMeter:
    """Computes and stores the sum and the last value"""

    def __init__(self,
                 init_factory: Callable = None,
                 init_value: Any = 0.,
                 init_count=0):
        self.init_factory: Callable = init_factory
        self.init_value = init_value

        self.reset(init_count)

    def reset(self, init_count=0):
        if self.init_factory:
            self.last = self.init_factory()
            self.sum = self.init_factory()
        else:
            self.last = self.init_value
            self.sum = self.init_value
        self.count = init_count

    def update(self, value, n=1):
        self.last = value
        self.sum += value
        self.count += n

    def get_average(self):
        return self.sum / self.count


def arr2str(a: np.ndarray, format_='e', ndigits=2) -> str:
    """convert ndarray of floats to a string expression.

    :param a:
    :param format_:
    :param ndigits:
    :return:
    """
    return np.array2string(
        a,
        formatter=dict(
            float_kind=(lambda x: f'{x:.{ndigits}{format_}}' if x != 0 else '0')
        )
    )


def print_to_file(fname: Union[str, Path], fn: Callable, args=None, kwargs=None):
    """ All `print` function calls in `fn(*args, **kwargs)`
      uses a text file `fname`.

    :param fname:
    :param fn:
    :param args: args for fn
    :param kwargs: kwargs for fn
    :return:
    """
    if fname:
        fname = Path(fname).with_suffix('.txt')

    if args is None:
        args = tuple()
    if kwargs is None:
        kwargs = dict()

    with (fname.open('w') if fname else open(os.devnull, 'w')) as file:
        with contextlib.redirect_stdout(file):
            fn(*args, **kwargs)
