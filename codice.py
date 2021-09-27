import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import PyOctaveBand
from math import sqrt
from hos import HOS

center, left, right = PyOctaveBand.getansifrequencies(fraction=3)
# center frequencies 1/3 octave
center = np.array(center)


def thirdoctave_from_signal(
    signal, sampling_rate, butterworth_filter_order=20, frq_limits=[12, 20000]
):
    """Compute the 1/3 octave spectrum of a signal in the time domain. Uses the
    library PyOctaveBand (https://github.com/jmrplens/PyOctaveBand).

    The spectrum returned is "normalized" (i.e. multiplied by `2/N`) by
    PyOctaveBand.

    :param np.ndarray signal: Input signal.
    :param float sampling_rate: The sampling rate at which the signal was
        collected.
    :param butterworth_filter_order: Order of the Butterworth filter used by
        PyOctaveBand, defaults to 20.
    :type butterworth_filter_order: int, optional
    :param frq_limits: The minimum and maximum frequencies considered in the
        spectrum, defaults to [12, 20000].
    :type frq_limits: list, optional
    :return: A tuple which contains:
        0. The center frequencies of the spectrum;
        1. The "normalized" 1/3 octave spectrum of the given signal.
    :rtype: tuple
    """
    spl, freq = PyOctaveBand.octavefilter(
        signal,
        fs=sampling_rate,
        fraction=3,
        order=butterworth_filter_order,
        limits=frq_limits,
    )
    return np.array(freq), np.array(spl)


# return the indexes of the given list of frequencies `fft_frq` which fall into
# the given range of frequencies `l` (left) and `r` (right).
def fft_bucket(l, r, fft_frq):
    indexes = []
    for idx, f in enumerate(fft_frq):
        if l <= f and f <= r:
            indexes.append(idx)
        if f > r:
            break
    return indexes


def thirdoctave_from_fft(frqs, fft_spectra):
    """Compute the 1/3 octave spectrum of a signal in the frequency domain. Uses
    the approach described at
    https://dsp.stackexchange.com/questions/46692/calculating-1-3-octave-spectrum-from-fft-dft

    The spectrum returned is NOT "normalized" (i.e. multiplied by `2/N`).

    :param np.ndarray frqs: The frequencies corresponding to the given DFT
        spectrum.
    :param np.ndarray: The DFT spectrum of the signal, where each position
        corresponds to the associated frequency in the array `frqs`.
    :return: A tuple which contains:
        0. The center frequencies of the spectrum;
        1. The 1/3 octave spectrum of the given signal.
    :rtype: tuple
    """
    fft_spectra = np.abs(np.array(fft_spectra))

    buckets = list(
        map(lambda tp: fft_bucket(tp[0], tp[1], frqs), zip(left, right))
    )

    amplitudes = []
    for b in buckets:
        amplitudes.append(sqrt(sum([pow(fft_spectra[i], 2) for i in b])))

    return center, amplitudes


def thirdoctave_from_hos(
    time, signal, avg=100, window_size=pow(2, 10), divide_by_N=True
):
    """Compute the 1/3 octave spectrum of a signal in the frequency domain. The
    approach used is the same of :func:`thirdoctave_from_fft`, but the signal
    is refined using the library HOS (UBE) before the computation of DFT.

    :param np.ndarray time: List of time instants corresponding to the sampling
        instants of the discrete signal.
    :param np.ndarray signal: Input signal.
    :param int avg: The number of averages used by UBE (HOS) to refine the
        spectrum.
    :param int window_size: The size of the windows used to split the signal.
        The number of windows is `len(signal) // window_size`, if the residual
        is not zero it is discarded.
    :param bool divide_by_N: If `True`, the spectrum is "normalized" (i.e.
        multiplied by `2/len(time)`) to ensure that it is effectively possible
        to recover the original signal using the inverse DFT.
    :return: A tuple which contains:
        0. The center frequencies of the spectrum;
        1. The 1/3 octave spectrum of the given signal.
    :rtype: tuple
    """
    h = HOS(time, signal, window_size=window_size)
    spectra, frqs = h.spectrum(avg)

    spectra = (
        2.0 / h.windowed_time.shape[1] if divide_by_N else 1
    ) * np.array(spectra)

    end_of_positive_frequencies = len(spectra) // 2 - 1
    spectra = spectra[:end_of_positive_frequencies]
    frqs = frqs[:end_of_positive_frequencies]

    return thirdoctave_from_fft(frqs, spectra)
