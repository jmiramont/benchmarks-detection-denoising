import numpy as np
from numpy import pi as pi
import scipy.signal as sg
from benchmark_demo.utilstf import hermite_fun
import string
# from matplotlib import pyplot as plt

class SignalBank:
    """
    Create a bank of signals. This class encapsulates the signal generation code, and returns different methods to generate signals
    as well as a dictionary of those methods that can be used later.
    """

    def __init__(self,N = 2**8):
        self.N = N
        self.generate_signal_dict()
        # self.signalDict2 = {
        #     'linearChirp': self.signal_linear_chirp,
        #     'cosChirp': self.signal_cos_chirp,
        #     'doubleCosChirp': self.signal_double_cos_chirp,
        #     'crossedLinearChirps': self.signal_crossing_linear_chirps,
        #     'dumpedCos': self.signal_dumped_cos,
        #     'sharpAttackCos': self.signal_sharp_attack,
        #     'multiComponentHarmonic' : self.signal_mc_harmonic,
        #     'multiComponentPureTones': self.signal_mc_pure_tones,
        #     'syntheticMixture': self.signal_synthetic_mixture,
        #     'hermiteFunction': self.signal_hermite_function,
        # }
    

    def generate_signal_dict(self):
        campos = dir(self)
        fun_names = [ fun_name for fun_name in campos if fun_name.startswith('signal_')]
        signal_ids = [string.capwords(fun_name[7::], sep = '_').replace('_','') for fun_name in fun_names]
        
        self.signalDict = dict()
        for i, signal_id in enumerate(signal_ids):
            self.signalDict[signal_id] = getattr(self, fun_names[i])
        return self.signalDict

    def get_signal_id(self):
        return self.SignalDict.keys()


    def signal_linear_chirp(self, a=0.25, b=0.125, instfreq = False):
        N = self.N
        t = np.arange(N)/N

        tmin = 1*int(np.sqrt(N))
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


    def signal_mc_crossing_chirps(self):
        N = self.N
        chirp1 = self.signal_linear_chirp(a = -0.25, b = 0.5 - 0.125)
        chirp2 = self.signal_linear_chirp(a = 0.25, b = 0.125)
        return chirp1 + chirp2


    def signal_mc_pure_tones(self, ncomps=5, max_freq=0.43, a1=0, b1 = 0.12):
        N = self.N
        k = 1
        aux = np.zeros((N,))
        
        for i in range(ncomps):
            chirp, instf, tmin, _ = self.signal_linear_chirp(a = a1*(i+1), b = b1*(i+1), instfreq=True)
            if instf[0] >= max_freq:
                break

            idx = np.where(instf < max_freq)[0] + tmin -1
            tukwin = sg.windows.tukey(idx.shape[0],0.25)
            chirp[idx] = chirp[idx]*tukwin
            idx = np.where(instf >= max_freq)[0] + tmin -1
            chirp[idx] = tukwin[-1]  
        
            aux += chirp
        return aux

    def signal_mc_harmonic(self, ncomps=5, max_freq=0.43, a1=0.07, b1=0.07):
       return self.signal_mc_pure_tones(ncomps=ncomps, max_freq=max_freq, a1=a1, b1=b1)


    def signal_tone_dumped(self):
        N = self.N
        eps = 1e-6
        t = np.arange(N)+eps
        c = 1/N/10
        prec = 1e-1 # Precision at sample N for the envelope.
        alfa = -np.log(prec*N/((N-c)**2))/N
        e = np.exp(-alfa*t)*((t-c)**2/t)
        e[0] = 0
        chirp = self.signal_linear_chirp(a = 0, b = 0.25)
        return e*chirp


    def signal_tone_sharp_attack(self):
        N = self.N
        dumpcos = self.signal_tone_dumped()
        indmax = np.argmax(dumpcos)
        dumpcos[0:indmax] = 0
        return dumpcos    


    def signal_cos_chirp(self, omega = 1.5, a1=1, f0=0.25, a2=0.125):
        N = self.N
        t = np.arange(N)/N
        tmin = int(np.sqrt(N))
        tmax = N-tmin
        Nsub = tmax-tmin
        tsub = np.arange(Nsub)
        instf = f0 + a2*np.cos(2*pi*omega*tsub/Nsub - pi*omega)    
        phase = np.cumsum(instf)
        x = a1*np.cos(2*pi*phase)*sg.tukey(Nsub,0.25)     
        signal = np.zeros((N,))
        signal[tmin:tmax] = x
        return signal


    def signal_mc_double_cos_chirp(self):
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


    def signal_mc_cos_plus_tone(self):
        N = self.N
        tmin = int(np.sqrt(N))
        tmax = N-tmin
        Nsub = tmax-tmin
        tsub = np.arange(Nsub)
        omega1 = 1.5
        omega2 = 1.8
        instf1 = 0.36 + 0.04*np.cos(2*pi*omega1*tsub/Nsub - pi*omega1)    
        instf2 = 0.23 + 0.05*np.cos(2*pi*omega2*tsub/Nsub - pi*omega2)
        instf3 = 0.1 * np.ones((Nsub,))    
        phase1 = np.cumsum(instf1)
        phase2 = np.cumsum(instf2)
        phase3 = np.cumsum(instf3)
        x = np.cos(2*pi*phase1) + np.cos(2*pi*phase2) + np.cos(2*pi*phase3)
        x = x*sg.tukey(Nsub,0.25)     
        signal = np.zeros((N,))
        signal[tmin:tmax] = x
        return signal

    
    def signal_mc_synthetic_mixture(self):
        N = self.N
        signal = np.zeros((N,))
        tmin = int(np.sqrt(N))
        tmax = N-tmin
        Nsub = tmax-tmin
        # print(Nchirp)
        # print(N-Nchirp)
        imp_loc_1 = 2*tmin
        imp_loc_2 = 3*tmin
        
        Nchirp = Nsub-4*tmin
        t = np.arange(Nchirp)
            
        chirp1 = np.cos(2*pi*0.1*t)
        b = 0.12
        a = (0.3-0.12)/Nchirp/2
        chirp2 = np.cos(2*pi*(a*t**2 + b*t))
        
        signal[imp_loc_1] = 10
        signal[imp_loc_2] = 10

        instf = 0.35 + 0.05*np.cos(2*pi*1.25*t/Nchirp + pi)
        coschirp = np.cos(2*pi*np.cumsum(instf))
        signal[4*tmin:4*tmin+Nchirp] = chirp1+chirp2+coschirp
        # signal[N-Nchirp:N] = coschirp
        return signal

    
    def signal_hermite_function(self, order = 30, t0 = 0.5, f0 = 0.25):
        N = self.N
        t0 = int(N*t0)
        t = np.arange(N)-t0
        return hermite_fun(N, order, t=t)*np.cos(2*pi*f0*t)

    
    def signal_hermite_elipse(self, order = 30, t0 = 0.5, f0 = 0.25):
        N = self.N
        t0 = int(N*t0)
        t = np.arange(N)-t0
        return hermite_fun(N, order, t=t, T = 2*np.sqrt(N))*np.cos(2*pi*f0*t)


    def signal_mc_triple_impulse(self, Nimpulses = 3):
        N = self.N
        tmin = int(np.sqrt(N))
        tmax = N-tmin
        Nsub = tmax-tmin
        dloc = Nsub/(Nimpulses+1)
        impulses = np.zeros((Nsub,))
        signal = np.zeros((N,))
        for i in range(Nimpulses):
            impulses[int((i+1)*dloc -1 )] = 10
        
        signal[tmin:tmax] = impulses
        return signal


    def signal_mc_triple_impulse_2(self, Nimpulses = 3):
        N = self.N
        tmin = int(np.sqrt(N))
        tmax = N-tmin
        Nsub = tmax-tmin
        dloc = Nsub/(Nimpulses+1)
        impulses = np.zeros((Nsub,))
        signal = np.zeros((N,))
        for i in range(Nimpulses):
            impulses[int((i+1)*dloc -1 )] = 10*(i+1)
        
        signal[tmin:tmax] = impulses
        return signal


    def signal_exp_chirp(self, finit=0.1, fend=0.43, exponent=2, r_instf=False):
        N = self.N
        tmin = 1*int(np.sqrt(N))
        tmax = N-tmin
        Nsub = tmax-tmin
        tsub = np.arange(Nsub)/Nsub
        instf =  finit*np.exp(np.log(fend/finit)*tsub**exponent)    
        phase = np.cumsum(instf)
        x = np.cos(2*pi*phase)
        signal = np.zeros((N,))
        signal[tmin:tmax] = x

        if r_instf:
            return signal, instf, tmin, tmax
        else:
            return signal

           
    def signal_mc_exp_chirps(self,  max_freq = 0.43):
        N = self.N
        signal = np.zeros((N,))
        aux = np.zeros((N,))
        exponents = [4, 3, 2]
        finits = [0.07, 0.18, 0.3]
        fends = [0.3, 0.8, 1.2]
        ncomps = len(fends)
        for i in range(ncomps):
            _, instf, tmin, tmax = self.signal_exp_chirp(finit=finits[i],
                                                         fend=fends[i],
                                                         exponent=exponents[i],
                                                         r_instf=True)    

            instf2 = instf
            
            if instf[0] >= max_freq:
                break
            
            instf2 = instf2[np.where(instf2 < max_freq)]
            tukwin = sg.windows.tukey(len(instf2),0.25)
            
            phase = np.cumsum(instf2)
            x = np.cos(2*pi*phase)
            tukwin = sg.windows.tukey(len(x),0.25)
            x = x*tukwin
            signal[tmin:tmin+len(x)] = x

            aux += signal
        return aux


    def signal_mc_modulated_tones(self):
        x1 = self.signal_cos_chirp(omega = 8, a1=1, f0=0.10, a2=0.04)
        x2 = self.signal_cos_chirp(omega = 6, a1=1, f0=0.25, a2=0.03)       
        x3 = self.signal_cos_chirp(omega = 4, a1=1, f0=0.40, a2=0.02)       
        # x4 = self.signal_cos_chirp(omega = 10, a1=1, f0=0.42, a2=0.05)              
        return x1+x2+x3


    def signal_mc_modulated_tones_2(self):
        x1 = self.signal_cos_chirp(omega = 5, a1=1.5, f0=0.10, a2=0.04)
        x2 = self.signal_cos_chirp(omega = 5, a1=1.2, f0=0.25, a2=0.03)       
        x3 = self.signal_cos_chirp(omega = 5, a1=1, f0=0.40, a2=0.02)       
        # x4 = self.signal_cos_chirp(omega = 10, a1=1, f0=0.42, a2=0.05)              
        return x1+x2+x3


    def signal_mc_synthetic_mixture_2(self):
        N = self.N
        t = np.arange(N)/N
        tmin = int(np.sqrt(N))
        tmax = N-tmin
        Nsub = tmax-tmin
        tsub = np.arange(Nsub)
        fmax = 0.43

        omega = 7
        f0 = 0.07 + 0.07*tsub/Nsub
        f1 = 0.11 + 0.25*tsub/Nsub
        f2 = 0.15 + 0.53*tsub/Nsub
        instf0 = f0 + 0.02*np.cos(2*pi*omega*tsub/Nsub - pi*omega)
        instf1 = f1 + 0.02*np.cos(2*pi*omega*tsub/Nsub - pi*omega)    
        instf2 = f2 + 0.02*np.cos(2*pi*omega*tsub/Nsub - pi*omega)    
        instf0 = instf0[np.where(instf0<fmax)]    
        instf1 = instf1[np.where(instf1<fmax)]    
        instf2 = instf2[np.where(instf2<fmax)]    
        
        phase0 = np.cumsum(instf0)
        phase1 = np.cumsum(instf1)
        phase2 = np.cumsum(instf2)

        signal = np.zeros((N,))
        x0 = np.zeros_like(signal)
        x1 = np.zeros_like(signal)
        x2 = np.zeros_like(signal)

        x0[tmin:tmin+len(phase0)] = np.cos(2*pi*phase0)*sg.tukey(len(phase0),0.25)
        x1[tmin:tmin+len(phase1)] = np.cos(2*pi*phase1)*sg.tukey(len(phase1),0.25) 
        x2[tmin:tmin+len(phase2)] = np.cos(2*pi*phase2)*sg.tukey(len(phase2),0.25) 

        signal = 2*x0+x1+0.5*x2 
        return signal


    def signal_mc_on_off_tones(self):
        N = self.N
        b1 = 0.12

        tmin = int(np.sqrt(N))
        tmax = N-tmin
        Nsub = tmax-tmin
        
        N3 = N//3
        N4 = N//4
        N7 = N//7
        N9 = N//9
        
        chirp1, instf, tmin, _ = self.signal_linear_chirp(a = 0, b = b1, instfreq=True)
        chirp2, instf, tmin, _ = self.signal_linear_chirp(a = 0, b = 2*b1, instfreq=True)
        chirp3, instf, tmin, _ = self.signal_linear_chirp(a = 0, b = 3*b1, instfreq=True)

        chirp1[0:2*N7] = 0
        chirp1[5*N7:-1] = 0
        chirp1[2*N7:5*N7] = chirp1[2*N7:5*N7]*sg.windows.tukey(3*N7,0.25)    

        chirp2[0:N9] = 0
        chirp2[4*N9:5*N9] = 0
        chirp2[8*N9:-1] = 0
        chirp2[N9:4*N9] = chirp2[N9:4*N9]*sg.windows.tukey(3*N9,0.25)    
        chirp2[5*N9:8*N9] = chirp2[5*N9:8*N9]*sg.windows.tukey(3*N9,0.25)    

        return chirp1+chirp2+chirp3



    def get_all_signals(self):
        signals = np.zeros((len(self.signalDict),self.N))
        for k, key in enumerate(self.signalDict):
            signals[k] = self.signalDict[key]()
        return signals

    
