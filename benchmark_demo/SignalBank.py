import numpy as np
from numpy import pi as pi
import scipy.signal as sg
# from benchmark_demo.utilstf import *
# from matplotlib import pyplot as plt

class SignalBank:
    """
    Create a bank of signals. This class encapsulates the signal generation code, and returns different methods to generate signals
    as well as a dictionary of those methods that can be used later.
    """

    def __init__(self,N = 2**8):
        self.N = N
        self.signalDict = {
            'linearChirp': self.linear_chirp,
            'cosChirp': self.cos_chirp,
            'crossedLinearChirps': self.crossing_linear_chirps,
            'dumpedCos': self.dumped_cos,
            'sharpAttackCos': self.sharp_attack_cos,
            'multiComponentHarmonic' : self.multi_component_harmonic,
            'syntheticMixture' : self.synthetic_mixture,
        }
        
    def get_signal_id(self):
        return self.SignalDict.keys()

    def linear_chirp(self, a=None, b=None, instfreq = False):
        N = self.N
        t = np.arange(N)/N

        tmin = int(np.sqrt(N))
        tmax = N-tmin
        
        Nsub = N-2*tmin
        
        if a is None:
            a = N/8
        if b is None:
            b = N/8


        tsub = t[tmin:tmax]
        phase = b*tsub + a*(tsub**2) - b*tmin
        instf = b + 2*a*tsub

        x = np.cos(2*pi*phase)*sg.tukey(Nsub) 
        signal = np.zeros((N,))
        signal[tmin:tmax] = x

        if instfreq:
            return signal, instf
        else:
            return signal

    def crossing_linear_chirps(self):
        N = self.N
        chirp1 = self.linear_chirp(a = -N/8, b = N/2 - N/8)
        chirp2 = self.linear_chirp(a = N/8, b = N/8)
        return chirp1 + chirp2

    def multi_component_harmonic(self, a1=0, b1 = None):
        N = self.N
        if b1 is None:
            b1 = 2*np.sqrt(N)
    
        k = 1
        aux = np.zeros((N,))
        chirp,instf = self.linear_chirp(a = a1, b = b1, instfreq=True)
        fn = instf[-1]

        while fn < (N/2-1*np.sqrt(N)):
            aux += chirp
            k += 1
            # print(k*b1) 
            chirp,instf = self.linear_chirp(a = a1*k, b = b1*k, instfreq=True)
            fn = instf[-1]
        return aux

    def dumped_cos(self):
        N = self.N
        eps = 1e-6
        t = np.arange(N)+eps
        c = 1/N/10
        prec = 1e-1 # Precision at sample N for the envelope.
        alfa = -np.log(prec*N/((N-c)**2))/N
        e = np.exp(-alfa*t)*((t-c)**2/t)
        e[0] = 0
        chirp = self.linear_chirp(a = 0, b = N/4)
        return e*chirp

    def sharp_attack_cos(self):
        N = self.N
        dumpcos = self.dumped_cos()
        indmax = np.argmax(dumpcos)
        dumpcos[0:indmax] = 0
        return dumpcos    

    def cos_chirp(self):
        N = self.N
        t = np.arange(N)/N

        tmin = int(np.sqrt(N))
        tmax = N-tmin
        Nsub = N-2*tmin
        tsub = t[tmin:tmax]
        
        phase = N/4*tsub + N/8*np.sin(2*pi*np.arange(Nsub)/Nsub)/2/pi - N/4*tmin
        # instf = N/4 + N/8*np.cos(2*pi*tsub)
        x = np.cos(2*pi*phase)*sg.tukey(Nsub)     

        signal = np.zeros((N,))
        signal[tmin:tmax] = x
        return signal
    
    def synthetic_mixture(self):
        N = self.N
        signal = np.zeros((N,))
        Nchirp = int(N*0.88)
        # print(Nchirp)
        # print(N-Nchirp)
        imp_loc_1 = int((N-Nchirp)/4)
        imp_loc_2 = int(2*(N-Nchirp)/3)
        t = np.arange(Nchirp)
            
        chirp1 = np.cos(2*pi*0.1*t)
        b = 0.12
        a = (0.3-0.12)/Nchirp/2
        chirp2 = np.cos(2*pi*(a*t**2 + b*t))
        
        signal[imp_loc_1] = 10
        signal[imp_loc_2] = 10

        instf = 0.35 + 0.05*np.cos(2*pi*1.25*t/Nchirp + pi)
        coschirp = np.cos(2*pi*np.cumsum(instf))
        signal[N-Nchirp:N] = chirp1+chirp2+coschirp
        # signal[N-Nchirp:N] = coschirp
        return signal

        

    
    def get_all_signals(self):
        signals = np.zeros((len(self.signalDict),self.N))
        for k, key in enumerate(self.signalDict):
            signals[k] = self.signalDict[key]()
        return signals

    


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from utilstf import *

    N = 512
    banco = SignalBank(N)
    # signal = banco.linearChirp(a = -N/8, b = N/2 - N/8)
    # signal = banco.linearChirp(a = N/8, b = N/8)
    # signal = banco.linearChirp(a = 0, b = N/4)
    # signal = banco.cosChirp()
    signal = banco.synthetic_mixture()
    # signal = banco.crossedLinearChirps()
    # signal = banco.multiComponentHarmonic(a1 = N/32)
    # signal = banco.dumpedCos()
    # signal = banco.sharpAttackCos()
    
    Sww, stft, pos, Npad = get_spectrogram(signal)

    fig, ax = plt.subplots(1,2)
    ax[0].plot(signal)
    ax[1].imshow(Sww, origin = 'lower')
    plt.show()