from scipy.signal import blackman
from scipy.fftpack import fft

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from sklearn import mixture


class HOS(object):


    def __init__(self, time, signal, window_size=1e10):

        if len(time) != len(signal):
            raise ValueError


        rem = len(time) % window_size
        if rem:
            print(
                'WARNING: input arrays are not divisible by window size.'
                ' Last {} elements are removed...'.format(rem))
            t = time[:-rem]
            x = signal[:-rem]
        else:
            t, x = time, signal

        self.dt = time[1]-time[0]
        self.window_size = window_size
        self.freq = np.fft.fftfreq(window_size, d=self.dt)
        self.blackman_window = blackman(window_size)
        self.windowed_time = np.array(np.split(t, np.floor(len(time)/(window_size))))
        self.windowed_sign = np.array(np.split(x, np.floor(len(signal)/(window_size))))
        self.windowed_fft = fft(
                (self.windowed_sign.T - np.average(self.windowed_sign, axis=1)).T *
                    self.blackman_window,
                axis=1)

    def spectrum(self, averages=10):

        return (np.mean(np.abs(self.windowed_fft[:averages, :]), axis=0),
               np.fft.fftfreq(self.window_size, d=self.dt))

    def ncoherence(self, averages=10, window_fraction=1/16, n=2):

        reduced_window_size = int(self.window_size * window_fraction)
        ranges = [reduced_window_size] * n

        term = np.zeros(shape=averages, dtype=np.complex)
        coherence = np.zeros(shape=ranges)

        Y = self.windowed_fft[:averages]
        for i in np.ndindex(*ranges):
            if all(i) is False: continue
            term[:] = np.prod(Y[:, i], axis=1) * np.conj(Y[:, sum(i)])
            coherence[i] = np.abs(np.mean(term))/(np.mean(np.abs(term)))
        #coherence = coherence.transpose((1, 0, 2)) # Same order of previous version

        print(coherence[1])
        return coherence

    def bicoherence(self, averages=10, window_fraction=1/16):
        return self.ncoherence(averages, window_fraction, n=2)

    def tricoherence(self, averages=10, window_fraction=1/16):
        return self.ncoherence(averages, window_fraction, n=3)

    def plot_bicoherence(self, bicoherence):

        f1, f2 = np.meshgrid(self.freq[:bicoherence.shape[0]], self.freq[:bicoherence.shape[0]])

        #surf = ax.plot_surface(f1, f2, bicoherence, rstride=1, cstride=1,cmap=cm.coolwarm)
        plot = plt.pcolor(f1, f2, bicoherence, cmap=cm.coolwarm)
        plt.colorbar(plot)
        plt.title('Bicoherence')
        plt.ylabel('$f_2$')
        plt.xlabel('$f_1$')
        plt.show()

    def plot_tricoherence(self, tricoherence):
        f1, f2, f3 = np.meshgrid(
            self.freq[:tricoherence.shape[0]],
            self.freq[:tricoherence.shape[0]],
            self.freq[:tricoherence.shape[0]])
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.scatter(f1.ravel(), f2.ravel(), f3.ravel(), marker='o', alpha=.5, s=np.exp(tricoherence.ravel()))
        plt.show()
