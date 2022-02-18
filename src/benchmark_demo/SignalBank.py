import numpy as np
from numpy import pi as pi
import scipy.signal as sg
from benchmark_demo.utilstf import hermite_fun
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
            'doubleCosChirp': self.double_cos_chirp,
            'crossedLinearChirps': self.crossing_linear_chirps,
            'dumpedCos': self.dumped_cos,
            'sharpAttackCos': self.sharp_attack_cos,
            'multiComponentHarmonic' : self.multi_component_harmonic,
            'multiComponentPureTones': self.multi_component_pure_tones,
            'syntheticMixture': self.synthetic_mixture,
            'hermiteFunction': self.hermite_function,
        }
        
    def get_signal_id(self):
        return self.SignalDict.keys()

    def linear_chirp(self, a=0.25, b=0.125, instfreq = False):
        N = self.N
        t = np.arange(N)/N

        tmin = int(np.sqrt(N))
        tmax = N-tmin        
        Nsub = tmax-tmin

        tsub = np.arange(Nsub)
        instf = b + a*tsub/Nsub
        phase = np.cumsum(instf)

        x = np.cos(2*pi*phase)*sg.tukey(Nsub,0.25) 
        signal = np.zeros((N,))
        signal[tmin:tmax] = x

        if instfreq:
            return signal, instf, tmin, tmax
        else:
            return signal


    def crossing_linear_chirps(self):
        N = self.N
        chirp1 = self.linear_chirp(a = -0.25, b = 0.5 - 0.125)
        chirp2 = self.linear_chirp(a = 0.25, b = 0.125)
        return chirp1 + chirp2


    def multi_component_pure_tones(self, ncomps=5, max_freq=0.43, a1=0, b1 = 0.12):
        N = self.N
        k = 1
        aux = np.zeros((N,))
        
        for i in range(ncomps):
            chirp, instf, tmin, _ = self.linear_chirp(a = a1*(i+1), b = b1*(i+1), instfreq=True)
            if instf[0] >= max_freq:
                break

            idx = np.where(instf < max_freq)[0] + tmin -1
            tukwin = sg.windows.tukey(idx.shape[0],0.25)
            chirp[idx] = chirp[idx]*tukwin
            idx = np.where(instf >= max_freq)[0] + tmin -1
            chirp[idx] = tukwin[-1]  
        
            aux += chirp
        return aux

    def multi_component_harmonic(self, ncomps=5, max_freq=0.43, a1=0.07, b1=0.07):
       return self.multi_component_pure_tones(ncomps=ncomps, max_freq=max_freq, a1=a1, b1=b1)


    def dumped_cos(self):
        N = self.N
        eps = 1e-6
        t = np.arange(N)+eps
        c = 1/N/10
        prec = 1e-1 # Precision at sample N for the envelope.
        alfa = -np.log(prec*N/((N-c)**2))/N
        e = np.exp(-alfa*t)*((t-c)**2/t)
        e[0] = 0
        chirp = self.linear_chirp(a = 0, b = 0.25)
        return e*chirp


    def sharp_attack_cos(self):
        N = self.N
        dumpcos = self.dumped_cos()
        indmax = np.argmax(dumpcos)
        dumpcos[0:indmax] = 0
        return dumpcos    


    def cos_chirp(self, omega = 1.5):
        N = self.N
        t = np.arange(N)/N
        tmin = int(np.sqrt(N))
        tmax = N-tmin
        Nsub = tmax-tmin
        tsub = np.arange(Nsub)
        instf = 0.25 + 0.125*np.cos(2*pi*omega*tsub/Nsub - pi*omega)    
        phase = np.cumsum(instf)
        x = np.cos(2*pi*phase)*sg.tukey(Nsub,0.25)     
        signal = np.zeros((N,))
        signal[tmin:tmax] = x
        return signal


    def double_cos_chirp(self):
        N = self.N
        t = np.arange(N)/N
        tmin = int(np.sqrt(N))
        tmax = N-tmin
        Nsub = tmax-tmin
        tsub = np.arange(Nsub)
        omega1 = 1.5
        omega2 = 1.8
        instf1 = 0.35 + 0.08*np.cos(2*pi*omega1*tsub/Nsub - pi*omega1)    
        instf2 = 0.15 + 0.08*np.cos(2*pi*omega2*tsub/Nsub - pi*omega2)    
        phase1 = np.cumsum(instf1)
        phase2 = np.cumsum(instf2)
        x = np.cos(2*pi*phase1) + np.cos(2*pi*phase2)
        x = x*sg.tukey(Nsub,0.25)     
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

    def hermite_function(self, order = 15, t0 = 0.5, f0 = 0.25):
        N = self.N
        t0 = int(N*t0)
        t = np.arange(N)-t0
        return hermite_fun(N, order, t=t)*np.cos(2*pi*f0*t)



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