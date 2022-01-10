import numpy as np
from numpy import pi as pi
from matplotlib import pyplot as plt
from utilstf import *

class SignalBank:
    """ Create a bank of signals"""

    def __init__(self,N = 2**8):
        self.N = N
        self.signalDict = {
            'linearChirp': self.linearChirp,
            'cosChirp': self.cosChirp,
            'crossedLinearChirps': self.crossedLinearChirps,
            'dumpedCos': self.dumpedCos,
            'sharpAttackCos': self.sharpAttackCos,
            'multiComponentHarmonic' : self.multiComponentHarmonic,
            # 'multiComponentHarmonic2' : lambda: self.multiComponentHarmonic(a1 = 4, b1 = 1.2*np.sqrt(N)),
        }
        

    def linearChirp(self, a=None, b=None, instfreq = False):
        N = self.N        
        tmin = int(np.sqrt(N))
        Nsub = N-2*tmin
        
        if a is None:
            a = Nsub/8
        if b is None:
            b = Nsub/8


        tsub = np.arange(Nsub)/Nsub
        phase = b*tsub + a*(tsub**2)
        instf = b + 2*a*tsub
        x = np.cos(2*pi*phase)*sg.tukey(Nsub) 
        signal = np.zeros((N,))
        signal[tmin:tmin+Nsub] = x
        if instfreq:
            return signal, instf
        else:
            return signal

    def crossedLinearChirps(self):
        N = self.N
        chirp1 = self.linearChirp(a = -N/8, b = N/2 - N/8)
        chirp2 = self.linearChirp(a = N/8, b = N/8)
        return chirp1 + chirp2

    def multiComponentHarmonic(self, a1=0, b1 = None):
        N = self.N
        if b1 is None:
            b1 = 2*np.sqrt(N)
    
        k = 1
        aux = np.zeros((N,))
        chirp,instf = self.linearChirp(a = a1, b = b1, instfreq=True)
        fn = instf[-1]

        while fn < (N/2-1*np.sqrt(N)):
            aux += chirp
            k += 1
            # print(k*b1) 
            chirp,instf = self.linearChirp(a = a1*k, b = b1*k, instfreq=True)
            fn = instf[-1]
        return aux

    def dumpedCos(self):
        N = self.N
        eps = 1e-6
        t = np.arange(N)+eps
        c = 1/N/10
        prec = 1e-1 # Precision at sample N for the envelope.
        alfa = -np.log(prec*N/((N-c)**2))/N
        e = np.exp(-alfa*t)*((t-c)**2/t)
        e[0] = 0
        chirp = self.linearChirp(a = 0, b = N/4)
        return e*chirp

    def sharpAttackCos(self):
        N = self.N
        dumpcos = self.dumpedCos()
        indmax = np.argmax(dumpcos)
        dumpcos[0:indmax] = 0
        return dumpcos    

    def cosChirp(self):
        N = self.N
        tmin = int(np.sqrt(N))
        Nsub = N-2*tmin
        tsub = np.arange(Nsub)/Nsub
        phase = N/4*tsub + N/8*np.sin(2*pi*tsub)/2/pi
        instf = N/4 + N/8*np.cos(2*pi*tsub)
        x = np.cos(2*pi*phase)    
        signal = np.zeros((N,))
        signal[tmin:tmin+Nsub] = x
        return signal
    
    def getAllSignals(self):
        signals = np.zeros((len(self.signalDict),self.N))
        for k, key in enumerate(self.signalDict):
            signals[k] = self.signalDict[key]()

        return signals

    # def createSignals(self, types = 'all'):
    #     for type in types:
            


if __name__ == '__main__':
    N = 1024
    banco = SignalBank(N)
    # signal1,_ = banco.linearChirp(a = -N/8, b = N/2 - N/8)
    # signal2,_ = banco.linearChirp(a = N/8, b = N/8)
    signal = banco.cosChirp()
    # signal = banco.crossedLinearChirps()
    # signal = banco.multiComponentHarmonic(a1 = N/32)
    # signal1 = banco.dumpedCos()
    # signal2 = banco.sharpAttackCos()
    
    pos, [Sww1, stft, x, y] = getSpectrogram(signal)
    # pos, [Sww2, stft, x, y] = getSpectrogram(signal2)

    fig, ax = plt.subplots(1,2)
    ax[0].plot(signal)
    ax[1].imshow(Sww1)

    plt.show()