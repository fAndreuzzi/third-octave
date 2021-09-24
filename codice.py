import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import PyOctaveBand
from math import sqrt
from hos import HOS

center, left, right = PyOctaveBand.getansifrequencies(fraction=3)
center = np.array(center)


def thirdoctave_from_signal(
    signal, sampling_rate, butterworth_filter_order=20, frq_limits=[12, 20000]
):
    spl, freq = PyOctaveBand.octavefilter(
        signal,
        fs=sampling_rate,
        fraction=3,
        order=butterworth_filter_order,
        limits=frq_limits,
    )
    return np.array(freq), np.array(spl)


def fft_bucket(l, r, fft_frq):
    indexes = []
    for idx, f in enumerate(fft_frq):
        if l <= f and f <= r:
            indexes.append(idx)
        if f > r:
            break
    return indexes


# only positive frequencies
def thirdoctave_from_fft(frqs, fft_spectra):
    fft_spectra = np.abs(np.array(fft_spectra))

    buckets = list(
        map(lambda tp: fft_bucket(tp[0], tp[1], frqs), zip(left, right))
    )

    amplitudes = []
    for b in buckets:
        amplitudes.append(sqrt(sum([pow(fft_spectra[i], 2) for i in b])))

    return center, amplitudes


def thirdoctave_from_hos(time, signal, avg=100, window_size=pow(2, 10), divide_by_N=True):
    h = HOS(time, signal, window_size=window_size)
    spectra, frqs = h.spectrum(avg)

    spectra = (2.0 / h.windowed_time.shape[1] if divide_by_N else 1) * np.array(spectra)

    end_of_positive_frequencies = len(spectra) // 2 - 1
    spectra = spectra[:end_of_positive_frequencies]
    frqs = frqs[:end_of_positive_frequencies]

    return thirdoctave_from_fft(frqs, spectra)
