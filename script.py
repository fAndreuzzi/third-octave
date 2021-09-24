import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from PyOctaveBand import getansifrequencies
from math import sqrt
from hos import HOS
from codice import *

fs = pow(2,18)
duration = 5  # In seconds
N = np.round(fs * duration)

x = np.arange(np.round(fs * duration)) / fs

def signal(x):
    # Signal with 6 frequencies
    f1, f2, f3, f4, f5, f6 = 20, 100, 500, 2000, 4000, 15000
    # Multi Sine wave signal
    return 100 \
        * (np.sin(2 * np.pi * f1 * x)
           + np.sin(2 * np.pi * f2 * x)
           + np.sin(2 * np.pi * f3 * x)
           + np.sin(2 * np.pi * f4 * x)
           + np.sin(2 * np.pi * f5 * x)
           + np.sin(2 * np.pi * f6 * x))

print(thirdoctave_from_signal(signal(x), fs))

f = fft(signal(x))
frq = fftfreq(N, 1/fs)

print(thirdoctave_from_fft(frq[:N//2-1], f[:N//2-1]))
print(thirdoctave_from_hos(x, signal(x)))
