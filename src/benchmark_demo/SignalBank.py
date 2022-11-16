import numpy as np
from numpy import pi as pi
import scipy.signal as sg
from benchmark_demo.utilstf import hermite_fun, get_stft, reconstruct_signal_2
import string
# from matplotlib import pyplot as plt


class Signal(np.ndarray):

    def __new__(subtype, array, instf=None, dtype=float, buffer=None, offset=0,
                strides=None, order=None):

        shape = array.shape
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = super().__new__(subtype, shape, dtype,
                              buffer, offset, strides, order)
        # set the new 'info' attribute to the value passed
        # obj._ncomps = ncomps
        obj[:] = array[:]
        obj._comps = [array.copy(), ]
        if instf is None:
            obj._instf = [np.zeros_like(array),]
        else:
            obj._instf = [instf.copy(),]    

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        self._comps = getattr(obj, '_comps', [obj, ])
        self._instf = getattr(obj, '_instf', list()) 
        
        # [np.zeros_like(self._comps[0]),]

        # self._ncomps = getattr(obj,'_ncomps', None)
        # We do not need to return anything
    
    # def __init__(self, array=None, instf=None):
    #     self.comps = list()
    #     self.comps.append(array)

    #     if instf is None:  
    #         self.instf = np.zeros_like(array)
    #     else:
    #         self.instf = instf

    # def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    #     args = []
        
    #     for i, input_ in enumerate(inputs):
    #         if isinstance(input_, Signal):
    #             args.append(input_.view(np.ndarray))
    #         else:
    #             args.append(input_)

    #     results = super().__array_ufunc__(ufunc, method, *args, **kwargs)

    #     return results
    
    def __add__(self,x):
        obj = super().__add__(x)

        if isinstance(x, Signal):
            if len(x) == len(x.comps[0]):
               obj = obj.view(Signal)
               obj._comps = []
               obj._instf = []
               for cp, instf in zip([*self.comps, *x.comps],
                                                    [*self.instf, *x.instf]):
                    obj.add_comp(cp, instf=instf)

        return obj

    @property
    def ncomps(self):
        return len(self._comps)

    @property
    def comps(self):
        return self._comps

    @property
    def instf(self):
        return self._instf    
    
    def add_comp(self, new_comp, **kwargs):
        self._comps.append(new_comp)
        if 'instf' in kwargs.keys():
            self._instf.append(kwargs['instf'])

    def add_instf(self, new_instf, **kwargs):
        self._instf.append(new_instf)        


class SignalBank:
    """
    Create a bank of signals. This class encapsulates the signal generation code,
    and returns different methods to generate signals as well as a dictionary of those 
    methods that can be used later. Methods starting with "signal" generate
    monocomponent signals. Methods starting with "signal_mc" generate multicomponent 
    signals.

    Both types of signals al generated with a length "N" passed as input parameter at
    the moment of instantiation. Signals are separated at least N^0.5 samples from the
    borders of the time-frequency plane in order to reduce border effects.
    
    Methods
    -------
        def check_frec_margins(self, instf):
            Check that the instantaneous frequency (if available) of a generated signal
            is withing certain margins to avoid aliasing and border effects.

        def generate_signal_dict(self):
            This function is used by the class constructor to generate a dictionary of 
            signals. The keys of this dictionary are the name of the signals or
            "signal_id" that is used to indicate the benchmark which signals use to 
            compare methods. 

        def get_signal_id(self):
            Get the keys of the dictionary of signals generated by the function 
            "generate_signal_dict()" when needed.

        def signal_linear_chirp(self, a=None, b=None, instfreq = False):
            Returns a linear chirp, the instantaneous frequency of which is a linear
            function with slope "a" and initial normalized frequency "b".

        def signal_mc_crossing_chirps(self):
            Returns a multi component signal with two chirps crossing, i.e. two chirps 
            whose instantaneous frequency coincide in one point of the time frequency
            plane.
   
        def signal_mc_pure_tones(self, ncomps=5, a1=None, b1=None):
            Generates a multicomponent signal comprising several pure tones harmonically
            separated, i.e. tones are ordered from lower to higher frequency and each
            one has an instantaneous frequency that is an entire multiple of that of the
            previous tone.
        
        def signal_mc_multi_linear(self, ncomps=5):
            Generates a multicomponent signal with multiple linear chirps.
   
        def signal_tone_dumped(self):
            Generates a dumped tone whose normalized frequency is 0.25.
    
        def signal_tone_sharp_attack(self):
            Generates a dumped tone that is modulated with a rectangular window.

        def signal_cos_chirp(self, omega=1.5, a1=1, f0=0.25, a2=0.125, checkinstf=True):
            Generates a cosenoidal chirp, the instantenous frequency of which is given 
            by the formula: "f0 + a1*cos(2*pi*omega)", and the maximum amplitude of
            which is determined by "a2".

        def signal_mc_double_cos_chirp(self):
            Generates a multicomponent signal with two cosenoidal chirps.

        def signal_mc_cos_plus_tone(self):
            Generates a multicomponent signal comprised by two cosenoidal chirps and a
            single tone.

        def signal_mc_synthetic_mixture(self):
            Generates a multicomponent signal with different types of components.

        def signal_hermite_function(self, order = 18, t0 = 0.5, f0 = 0.25):
            Generates a round hermite function of a given order. The spectrogram of
            Hermite functions are given by an annular ridge in the time frequency plane,
            the center of which is given by (t0,f0).

        def signal_hermite_elipse(self, order = 30, t0 = 0.5, f0 = 0.25):
            Generates a non-round Hermite function of a given order. The spectrogram of
            Hermite functions are given by an elipsoidal ridge in the time frequency
            plane the center of which is given by (t0,f0).

        def signal_mc_triple_impulse(self, Nimpulses = 3):
            Generates three equispaced impulses in time.

        def signal_mc_impulses(self, Nimpulses = 7):
            Generates equispaced impulses in time.

        def signal_exp_chirp(self, finit=None, fend=None, exponent=2, r_instf=False):
            Generates an exponential chirp.

        def signal_mc_exp_chirps(self):
            Generates a multicomponent signal comprising three exponential chirps.

        def signal_mc_multi_cos(self):
            Generates a multicomponent signal comprising three cosenoidal chirps with
            different frequency modulation parameters.

        def signal_mc_synthetic_mixture_2(self):
            Generates a multicomponent signal with different types of components.

        def signal_mc_on_off_tones(self):
            Generates a multicomponent signal comprising components that "born" and 
            "die" at different times.

        def signal_mc_synthetic_mixture_3(self):
            Generates a multicomponent signal with different types of components.

        def get_all_signals(self):
            Returns an array of shape [K,N] where K is the number of signals generated
            by this signal bank, and N is the length of the signals.    
    """
    
    def __init__(self, N = 2**8, Nsub = None, return_signal=False):
        """ Builds a dictionary of functions that return multiple signals.
        Args:
            N (int, optional): Length of the signals. Defaults to 2**8.
            Nsub (int, optional): Generates signals of length Nsub,
            then zero pads the vector to reach length N. Defaults to 2**8.
            return_signal (bool, optional): If True, functions will return a Signal
            object, that encapsulates more information of the signal such as the number
            of components, each individual component and their instantaneous frequency.
        """
        self.return_signal = return_signal

        self.N = N
        if Nsub is None: 
            self.tmin = int(np.sqrt(N))
            self.tmax = self.N - self.tmin
            self.Nsub = self.tmax-self.tmin
        else:                
            self.Nsub = Nsub
            self.tmin = (self.N-self.Nsub)//2
            self.tmax = self.tmin+Nsub
                    
        # self.fmin = 1.0*np.sqrt(N)/N
        # self.fmax = 0.5-self.fmin
        # self.fmid = (self.fmax-self.fmin)/2 + self.fmin

        self.fmin = 0.07
        self.fmax = 0.5-self.fmin
        self.fmid = 0.25

        # print(self.fmin)
        # print(self.fmax)

        self.generate_signal_dict()

    # # Create a decorator to cast the output of functions, so that we can activate and
    # # deactivate the use of the Signal class, and use just numpy arrays instead.
    # def modify_output(self, signal_generation_function):
    #     def wrapper(*args,**kwargs):
    #         signal = signal_generation_function(*args,**kwargs)
    #         if not self.return_signal:
    #             return signal.view(np.ndarray)
    #         else:
    #             return signal    
    #     return wrapper

    # TODO
    def check_inst_freq(self, instf):
        """Check that the instantaneous frequency (if available) of a generated signal
        is withing certain margins to avoid aliasing and border effects.

        Args:
            instf (numpy.ndarray): Instantaneous frequency of a signal.
        """

        # if np.all(instf<=self.fmax):
        #     print('Warning: instf>fmax')
        # if np.all(instf>=self.fmin):
        #     print('Warning: instf<fmin')
        return True

    def generate_signal_dict(self):
        """ This function is used by the class constructor to generate a dictionary of 
        signals. The keys of this dictionary are the name of the signals or "signal_id"
        that is used to indicate the benchmark which signals use to compare methods. 

        Returns:
            dict: Dictionary of functions that returns a signal when called.
        """

        campos = dir(self)
        fun_names = [ fun_name for fun_name in campos if fun_name.startswith('signal_')]
        signal_ids = [string.capwords(fun_name[7::], sep = '_').replace('_','') 
                    for fun_name in fun_names]
        
        self.signalDict = dict()
        for i, signal_id in enumerate(signal_ids):
            self.signalDict[signal_id] = getattr(self, fun_names[i])
        return self.signalDict


    def get_signal_id(self):
        """ Get the keys of the dictionary of signals generated by the function 
        "generate_signal_dict()" when needed.

        Returns:
            tuple: Tuple with the keys of a dictionary of signals. 
        """

        return self.SignalDict.keys()

    def get_all_signals(self):
        """Returns an array of shape [K,N] where K is the number of signals generated
        by this signal bank, and N is the length of the signals.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """

        signals = np.zeros((len(self.signalDict),self.N))
        for k, key in enumerate(self.signalDict):
            signals[k] = self.signalDict[key]()
        return signals


# Monocomponent signals --------------------------------------------------------

    def _signal_linear_chirp(self, a=None, b=None, instfreq = False):
        """Returns a linear chirp, the instantaneous frequency of which is a linear
        function with slope "a" and initial normalized frequency "b".


        Args:
            a (int, optional): Slope of the instantaneous frequency. Defaults to None.
            b (int, optional): Initial instantaneous frequency. Defaults to None.
            instfreq (bool, optional): When True, returns a vector with the
            instantaneous frequency. Defaults to False.

        Returns:
            list or ndarray: If input parameter "instfreq" is True, returns the a list
            of ndarray type objects with the signal and its instantaneous frequency.
        """

        N = self.N

        if a is None:
            a = self.fmax-self.fmin
        if b is None:
            b = self.fmin

        tmin = self.tmin
        tmax = self.tmax      
        Nsub = self.Nsub

        tsub = np.arange(Nsub)
        instf = b + a*tsub/Nsub

        if not instfreq:
            self.check_inst_freq(instf)

        phase = np.cumsum(instf)

        x = np.cos(2*pi*phase)*sg.tukey(Nsub,0.25) 
        signal = np.zeros((N,))
        signal[tmin:tmax] = x

        emb_instf = np.zeros_like(signal)
        emb_instf[tmin:tmax] = instf

        # Cast to Signal class.
        signal = Signal(array=signal, instf=emb_instf)
    
        if instfreq:
            return signal, instf, tmin, tmax
        else:
            return signal
      
    def _signal_tone_dumped(self):
        """Generates a dumped tone whose normalized frequency is 0.25.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """

        N = self.N
        eps = 1e-6
        t = np.arange(N)+eps
        c = 1/N/10
        prec = 1e-1 # Precision at sample N for the envelope.
        alfa = -np.log(prec*N/((N-c)**2))/N
        e = np.exp(-alfa*t)*((t-c)**2/t)
        e[0] = 0
        chirp = self._signal_linear_chirp(a = 0, b = 0.25)
        return e*chirp

    def _signal_tone_sharp_attack(self):
        """Generates a dumped tone that is modulated with a rectangular window.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """

        N = self.N
        dumpcos = self.signal_tone_dumped()
        indmax = np.argmax(dumpcos)
        dumpcos[0:indmax] = 0
        return dumpcos    

    def _signal_exp_chirp(self, finit=None, fend=None, exponent=2, r_instf=False):
        """Generates an exponential chirp.

        Args:
            finit (float, optional): Initial normalized frequency. Defaults to None.
            fend (float, optional): End normalized frequency. Defaults to None.
            exponent (int, optional): Exponent. Defaults to 2.
            r_instf (bool, optional): When True returns the instantaneous frequency
            along with the signal. Defaults to False.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """
        
        N = self.N
        tmin = self.tmin
        tmax = self.tmax      
        Nsub = self.Nsub
        tsub = np.arange(Nsub)/Nsub

        if finit is None:
            finit=1.5*self.fmin

        if fend is None:    
            fend=self.fmax

        instf =  finit*np.exp(np.log(fend/finit)*tsub**exponent)

        if not r_instf:    
            self.check_inst_freq(instf)

        phase = np.cumsum(instf)
        x = np.cos(2*pi*phase)
        signal = np.zeros((N,))
        signal[tmin:tmax] = x*sg.windows.tukey(Nsub,0.25)

        emb_instf = np.zeros_like(signal)
        emb_instf[tmin:tmax] = instf

        # Cast to Signal class.
        signal = Signal(array=signal, instf=emb_instf)

        if r_instf:
            return signal, instf, tmin, tmax
        else:
            return signal

    def _signal_cos_chirp(self, omega = 1.2, a1=0.5, f0=0.25, a2=0.125, checkinstf = True):
        """Generates a cosenoidal chirp, the instantenous frequency of which is given by
        the formula: "f0 + a1*cos(2*pi*omega)", and the maximum amplitude of which is
        determined by "a2".

        Args:
            omega (float, optional): Frequency of the instantaneous frequency.
            Defaults to 1.5.
            a1 (int, optional): Amplitude of the frequency modulation Defaults to 1.
            f0 (float, optional): Central frequency. Defaults to 0.25.
            a2 (float, optional): Amplitude of the signal. Defaults to 0.125.
            checkinstf (bool, optional): If True checks that dhe instantaneous frequency
            of the signal is within the limits. Defaults to True.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """

        N = self.N
        tmin = self.tmin
        tmax = self.tmax      
        Nsub = self.Nsub
        tsub = np.arange(Nsub)
        instf = f0 + a2*np.cos(2*pi*omega*tsub/Nsub - pi*omega)

        if checkinstf:
            self.check_inst_freq(instf)    

        phase = np.cumsum(instf)
        x = a1*np.cos(2*pi*phase)*sg.tukey(Nsub,0.25)     
        signal = np.zeros((N,))
        signal[tmin:tmax] = x

        emb_instf = np.zeros_like(signal)
        emb_instf[tmin:tmax] = instf

        # Cast to Signal class.
        signal = Signal(array=signal, instf=emb_instf)

        return signal

# Output monocomponent signals -------------------------------------------------
    def signal_linear_chirp(self, a=None, b=None, instfreq = False):
        """Returns a linear chirp, the instantaneous frequency of which is a linear
        function with slope "a" and initial normalized frequency "b".


        Args:
            a (int, optional): Slope of the instantaneous frequency. Defaults to None.
            b (int, optional): Initial instantaneous frequency. Defaults to None.
            instfreq (bool, optional): When True, returns a vector with the
            instantaneous frequency. Defaults to False.

        Returns:
            list or ndarray: If input parameter "instfreq" is True, returns the a list
            of ndarray type objects with the signal and its instantaneous frequency.
        """

        signal = self._signal_linear_chirp(a=a, b=b, instfreq=instfreq)
        if not self.return_signal:
            return signal.view(np.ndarray)
        return signal


    def signal_tone_dumped(self):
        """Generates a dumped tone whose normalized frequency is 0.25.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """

        N = self.N
        eps = 1e-6
        t = np.arange(N)+eps
        c = 1/N/10
        prec = 1e-1 # Precision at sample N for the envelope.
        alfa = -np.log(prec*N/((N-c)**2))/N
        e = np.exp(-alfa*t)*((t-c)**2/t)
        e[0] = 0
        chirp = self._signal_linear_chirp(a = 0, b = 0.25)
        signal = e*chirp
        
        if not self.return_signal:
            return signal.view(np.ndarray)

        return signal

    def signal_tone_sharp_attack(self):
        """Generates a dumped tone that is modulated with a rectangular window.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """

        N = self.N
        signal = self.signal_tone_dumped()
        indmax = np.argmax(signal)
        signal[0:indmax] = 0

        if not self.return_signal:
            return signal.view(np.ndarray)

        return signal

    def signal_exp_chirp(self, finit=None, fend=None, exponent=2, r_instf=False):
        """Generates an exponential chirp.

        Args:
            finit (float, optional): Initial normalized frequency. Defaults to None.
            fend (float, optional): End normalized frequency. Defaults to None.
            exponent (int, optional): Exponent. Defaults to 2.
            r_instf (bool, optional): When True returns the instantaneous frequency
            along with the signal. Defaults to False.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """
        signal = self._signal_exp_chirp(finit=finit, fend=fend, exponent=exponent, r_instf=r_instf)
        if not self.return_signal:
            return signal.view(np.ndarray)
        return signal

    def signal_cos_chirp(self, omega = 1.2, a1=0.5, f0=0.25, a2=0.125, checkinstf = True):
        """Generates a cosenoidal chirp, the instantenous frequency of which is given by
        the formula: "f0 + a1*cos(2*pi*omega)", and the maximum amplitude of which is
        determined by "a2".

        Args:
            omega (float, optional): Frequency of the instantaneous frequency.
            Defaults to 1.5.
            a1 (int, optional): Amplitude of the frequency modulation Defaults to 1.
            f0 (float, optional): Central frequency. Defaults to 0.25.
            a2 (float, optional): Amplitude of the signal. Defaults to 0.125.
            checkinstf (bool, optional): If True checks that dhe instantaneous frequency
            of the signal is within the limits. Defaults to True.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """
        signal = self._signal_cos_chirp(omega = omega, a1=a1, f0=f0, a2=a2, checkinstf=checkinstf)
        if not self.return_signal:
            return signal.view(np.ndarray)
        return signal


# Multicomponent signals here --------------------------------------------------

    def signal_mc_parallel_chirps(self):
        comp1 = self._signal_linear_chirp(a=0.1, b=0.15)
        comp2 = self._signal_linear_chirp(a=0.1, b=0.25)
        signal = comp1+comp2

        if not self.return_signal:
            return signal.view(np.ndarray)

        return signal

    def signal_mc_parallel_chirps_unbalanced(self):
        comp1 = self._signal_linear_chirp(a=0.1, b=0.15, instfreq = False)
        comp2 = self._signal_linear_chirp(a=0.1, b=0.25, instfreq = False)
        signal = comp1+0.2*comp2

        if not self.return_signal:
            return signal.view(np.ndarray)

        return signal

    def signal_mc_on_off_2(self):
        chirp1 = self._signal_linear_chirp(a=0.1, b=0.10, instfreq = False)
        chirp2 = self._signal_linear_chirp(a=0.1, b=0.20, instfreq = False)
        chirp3 = self._signal_linear_chirp(a=0.1, b=0.30, instfreq = False)

        Nsub = self.N
        N3 = Nsub//3
        N4 = Nsub//4
        N7 = Nsub//7
        N9 = Nsub//9

        chirp1[0:2*N7] = 0
        chirp1[5*N7:-1] = 0
        chirp1[2*N7:5*N7] = chirp1[2*N7:5*N7]*sg.windows.tukey(3*N7,0.25)    

        chirp2[0:N9] = 0
        chirp2[4*N9:5*N9] = 0
        chirp2[8*N9:-1] = 0
        chirp2[N9:4*N9] = chirp2[N9:4*N9]*sg.windows.tukey(3*N9,0.25)    
        chirp2[5*N9:8*N9] = chirp2[5*N9:8*N9]*sg.windows.tukey(3*N9,0.25) 

        signal = chirp1+chirp2+chirp3

        if not self.return_signal:
            return signal.view(np.ndarray)    

        return signal

    def signal_mc_crossing_chirps(self):
        """Returns a multi component signal with two chirps crossing, i.e. two chirps 
        whose instantaneous frequency coincide in one point of the time frequency plane.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """

        N = self.N
        
        a = self.fmax-self.fmin
        b = self.fmin
        
        chirp1 = self._signal_linear_chirp(a = -a, b = 0.5 - b)
        chirp2 = self._signal_linear_chirp(a = a, b = b)

        signal = chirp1 + chirp2

        if not self.return_signal:
            return signal.view(np.ndarray)
        
        return signal

    def signal_mc_pure_tones(self, ncomps=5, a1=None, b1=None):
        """Generates a multicomponent signal comprising several pure tones harmonically
        separated, i.e. tones are ordered from lower to higher frequency and each one
        has an instantaneous frequency that is an entire multiple of that of the
        previous tone.

        Args:
            ncomps (int, optional): Number of components. Defaults to 5.
            a1 (float, optional): Slope of chirps. Defaults to 0.
            b1 (float, optional): Frequency of the first tone. Defaults to None.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """

        N = self.N
        k = 1
        aux = np.zeros((N,))
        max_freq = self.fmax
        
        if a1 is None:
            a1=0
        if b1 is None:
            b1 = self.fmid/2
            if b1 < self.fmin:
                b1 = self.fmin

        signal = self._signal_linear_chirp(a = a1, b = b1)
        
        for i in range(1,ncomps):
            chirp = self._signal_linear_chirp(a = a1*(i+1), b = b1*(i+1))
            if np.max(chirp.instf[0]) >= max_freq:
                break
            signal = signal + chirp

        # for i in range(ncomps):
        #     chirp, instf, tmin, _ = self.signal_linear_chirp(a = a1*(i+1),
        #                                                      b = b1*(i+1),
        #                                                      instfreq=True)
        #     if instf[0] >= max_freq:
        #         break

        #     idx = np.where(instf < max_freq)[0] + tmin -1
        #     tukwin = sg.windows.tukey(idx.shape[0],0.5)
        #     chirp[idx] = chirp[idx]*tukwin
        #     idx = np.where(instf >= max_freq)[0] + tmin -1
        #     chirp[idx] = tukwin[-1]  
        
        #     aux += chirp
        
        if not self.return_signal:
            return signal.view(np.ndarray)
        
        return signal

    def signal_mc_multi_linear(self, ncomps=5):
        """Generates a multicomponent signal with multiple linear chirps.

        Args:
            ncomps (int, optional): Number of components. Defaults to 5.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """

        a1=self.fmin/2
        b1=self.fmin
        return self.signal_mc_pure_tones(ncomps=ncomps, a1=a1, b1=b1)

    def signal_mc_multi_linear_2(self, ncomps=5):
        """Generates a multicomponent signal with multiple linear chirps.

        Args:
            ncomps (int, optional): Number of components. Defaults to 5.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """

        a1=self.fmin/2
        b1=self.fmin/2
        return self.signal_mc_pure_tones(ncomps=ncomps, a1=a1, b1=b1)    

    def signal_mc_double_cos_chirp(self):
        """Generates a multicomponent signal with two cosenoidal chirps.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """

        N = self.N
        t = np.arange(N)/N
        tmin = self.tmin
        tmax = self.tmax      
        Nsub = self.Nsub
        tsub = np.arange(Nsub)
        omega1 = 1.5
        omega2 = 1.8
        
        instf1 = self.fmax-0.075 + 0.05*np.cos(2*pi*omega1*tsub/Nsub - pi*omega1)
        self.check_inst_freq(instf1)    
        instf2 = self.fmin+0.075 + 0.06*np.cos(2*pi*omega2*tsub/Nsub - pi*omega2)
        self.check_inst_freq(instf2)

        phase1 = np.cumsum(instf1)
        phase2 = np.cumsum(instf2)
        x = np.cos(2*pi*phase1) + np.cos(2*pi*phase2)
        x = x*sg.tukey(Nsub,0.25)     
        signal = np.zeros((N,))
        signal[tmin:tmax] = x
        return signal

    def signal_mc_cos_plus_tone(self):
        """Generates a multicomponent signal comprised by two cosenoidal chirps and a
        single tone.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """

        N = self.N
        tmin = self.tmin
        tmax = N-tmin
        Nsub = tmax-tmin
        tsub = np.arange(Nsub)
        omega1 = 1.5
        omega2 = 1.8

        instf1 = self.fmax-0.05 + 0.04*np.cos(2*pi*omega1*tsub/Nsub - pi*omega1)    
        self.check_inst_freq(instf1)
        instf2 = self.fmid + 0.04*np.cos(2*pi*omega2*tsub/Nsub - pi*omega2)
        self.check_inst_freq(instf2)
        instf3 = 1.8*self.fmin * np.ones((Nsub,))    
        self.check_inst_freq(instf3)
        
        phase1 = np.cumsum(instf1)
        phase2 = np.cumsum(instf2)
        phase3 = np.cumsum(instf3)
        x = np.cos(2*pi*phase1) + np.cos(2*pi*phase2) + np.cos(2*pi*phase3)
        x = x*sg.tukey(Nsub,0.25)     
        signal = np.zeros((N,))
        signal[tmin:tmax] = x
        return signal


    def signal_mc_multi_cos(self):
        """Generates a multicomponent signal comprising three cosenoidal chirps with
        different frequency modulation parameters.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """

        x1 = self._signal_cos_chirp(omega = 8, a1=1, f0=self.fmin+0.04, a2=0.03)
        x2 = self._signal_cos_chirp(omega = 6, a1=1, f0=self.fmid, a2=0.02)       
        x3 = self._signal_cos_chirp(omega = 4, a1=1, f0=self.fmax-0.03, a2=0.02)       
        # x4 = self.signal_cos_chirp(omega = 10, a1=1, f0=0.42, a2=0.05)              
        return x1+x2+x3

    def signal_mc_multi_cos_2(self):
        """Generates a multicomponent signal comprising three cosenoidal chirps with 
        different frequency modulation parameters.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """
        
        x1 = self._signal_cos_chirp(omega = 5, a1=1.5,   f0=self.fmin+0.04, a2=0.03)
        x2 = self._signal_cos_chirp(omega = 5, a1=1.2,   f0=self.fmid, a2=0.02)   
        x3 = self._signal_cos_chirp(omega = 5, a1=1,     f0=self.fmax-0.03, a2=0.02)       
        # x4 = self.signal_cos_chirp(omega = 10, a1=1, f0=0.42, a2=0.05)              
        return x1+x2+x3

    def signal_mc_synthetic_mixture_1(self):
        """Generates a multicomponent signal with different types of components.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """

        N = self.N
        t = np.arange(N)/N
        tmin = self.tmin
        tmax = N-tmin
        Nsub = tmax-tmin
        tsub = np.arange(Nsub)
        fmax = self.fmax

        omega = 7
        f0 = self.fmin + 0.07*tsub/Nsub
        f1 = 1.2*self.fmin + 0.25*tsub/Nsub
        f2 = 1.3*self.fmin + 0.53*tsub/Nsub

        instf0 = f0+0.02 + 0.02*np.cos(2*pi*omega*tsub/Nsub - pi*omega)
        instf1 = f1+0.02 + 0.02*np.cos(2*pi*omega*tsub/Nsub - pi*omega)    
        instf2 = f2+0.02 + 0.02*np.cos(2*pi*omega*tsub/Nsub - pi*omega)    
        instf0 = instf0[np.where(instf0<fmax)]    
        instf1 = instf1[np.where(instf1<fmax)]    
        instf2 = instf2[np.where(instf2<fmax)]    
        
        self.check_inst_freq(instf0)
        self.check_inst_freq(instf1)
        self.check_inst_freq(instf2)
        
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

        signal = x0+x1+x2 
        return signal


    def signal_mc_synthetic_mixture_2(self):
        """Generates a multicomponent signal with different types of components.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """
        def rect_window(N,ti,te):
            rw = np.zeros((N,))
            rw[ti:te] = 1
            return rw
            

        N = self.N
        t = np.arange(N)
        tmin = self.tmin
        tmax = N-tmin
        Nsub = tmax-tmin
        tsub = np.arange(Nsub)
        fmax = self.fmax
        fmin = self.fmin

        tt = Nsub//2
        t_init = tmin 

        f_init = (0.25-fmin)/2 + fmin
        f_end = fmax
        m = (f_end-f_init)/tt

        instf1 = (m*(t-t_init) + f_init)*rect_window(N,t_init,t_init+tt)
        phase1 = np.cumsum(instf1)
        x1 = np.cos(2*pi*phase1)*rect_window(N,t_init,t_init+tt)
        x1[t_init:t_init+tt]*=sg.tukey(tt,0.25)

        c = 1/tt/10
        prec = 1e-1 # Precision at sample N for the envelope.
        alfa = -np.log(prec*tt/((tt-c)**2))/tt
        e = np.exp(-alfa*np.arange(tt))*((np.arange(tt)-c)**2/np.arange(tt))
        e[0] = 0
        e /= np.max(e)


        t_init += tt//2
        instf2 = (m*(t-t_init) + f_init)*rect_window(N,t_init,t_init+tt)
        phase2 = np.cumsum(instf2)
        x2 = np.cos(2*pi*phase2)*rect_window(N,t_init,t_init+tt)
        x2[t_init:t_init+tt]*=sg.tukey(tt,0.25)*e

        t_init += tt//2
        instf3 = (m*(t-t_init) + f_init)*rect_window(N,t_init,t_init+tt)
        phase3 = np.cumsum(instf3)
        x3 = np.cos(2*pi*phase3)*rect_window(N,t_init,t_init+tt)
        x3[t_init:t_init+tt]*=sg.tukey(tt,0.25)*e[-1::-1]


        x4 = np.cos(2*pi*fmin*np.ones((N,))*t)*rect_window(N,tmin,tmax)
        x4[tmin:tmax] *= sg.tukey(Nsub,0.75)

        signal = x1 + x2 + x3 + x4
        return signal        


    def signal_mc_synthetic_mixture_3(self):
        """Generates a multicomponent signal with different types of components.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """

        N = self.N
        tmin = self.tmin
        tmax = self.tmax
        Nsub = self.Nsub
        tmid = Nsub//2
        # tmid = tmid +(tmax-tmid)//5
        tsub = np.arange(Nsub)
        
        sigma = 0.005
        
        instf1 = 1.5*self.fmin + 1*(tsub/Nsub-0.05)**2
        instf1 = instf1[np.where(instf1<self.fmax)]
        instf1 = instf1[np.where(self.fmin<instf1)]
        phase1 = np.cumsum(instf1)
        x = np.cos(2*pi*phase1)*sg.windows.tukey(len(phase1),0.25)

        instf2 = np.ones((Nsub,))*(self.fmid-self.fmin)/2
        instf2 = instf2[tmid::]
        phase2 = np.cumsum(instf2)
        x2 = np.cos(2*pi*phase2)*sg.windows.tukey(len(phase2),0.25)

        
        instf3 = np.ones((N,))*((self.fmid-self.fmin)/3 + self.fmid)
        phase3 = np.cumsum(instf3)
        tloc = 3*N//4
        x3 = 5*np.cos(2*pi*phase3)*np.exp(-np.pi*(np.arange(N)-tloc)**2/(N/8))
        
        
        signal = np.zeros((N,))
        signal[tmin:tmin+len(x)] = x
        signal[tmid:tmid+len(x2)] +=  x2 

        signal += x3


        return signal




    def signal_mc_on_off_tones(self):
        """Generates a multicomponent signal comprising components that "born" and "die"
        at different times.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """
        
        N = self.N
        b1 = self.fmid/2

        if b1 < self.fmin:
            b1 = self.fmin

        tmin = self.tmin
        tmax = N-tmin
        Nsub = tmax-tmin
        
        N3 = N//3
        N4 = N//4
        N7 = N//7
        N9 = N//9

        
        chirp1, instf, tmin, _ = self._signal_linear_chirp(a=0, b=b1, instfreq=True)
        chirp2, instf, tmin, _ = self._signal_linear_chirp(a=0, b=2*b1, instfreq=True)
        chirp3, instf, tmin, _ = self._signal_linear_chirp(a=0, b=3*b1, instfreq=True)

        chirp1[0:2*N7] = 0
        chirp1[5*N7:-1] = 0
        chirp1[2*N7:5*N7] = chirp1[2*N7:5*N7]*sg.windows.tukey(3*N7,0.5)    

        chirp2[0:N9] = 0
        chirp2[4*N9:5*N9] = 0
        chirp2[8*N9:-1] = 0
        chirp2[N9:4*N9] = chirp2[N9:4*N9]*sg.windows.tukey(3*N9,0.5)    
        chirp2[5*N9:8*N9] = chirp2[5*N9:8*N9]*sg.windows.tukey(3*N9,0.5)    

        return chirp1+chirp2+chirp3

# Other signals ----------------------------------------------------------------

    def signal_hermite_function(self, order = 18, t0 = 0.5, f0 = 0.25):
        """Generates a round hermite function of a given order. The spectrogram of
        Hermite functions are given by an annular ridge in the time frequency plane, the
        center of which is given by (t0,f0).

        Args:
            order (int, optional): Order of the Hermite function. Defaults to 18.
            t0 (float, optional): Time coordinate of the center of the spectrogram.
            Defaults to 0.5.
            f0 (float, optional): Frequency coordinate of the center of the spectrogram.
            Defaults to 0.25.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """
        
        N = self.N
        t0 = int(N*t0)
        t = np.arange(N)-t0
        return hermite_fun(N, order, t=t, T = np.sqrt(2*N))*np.cos(2*pi*f0*t)
  
    def signal_hermite_elipse(self, order = 30, t0 = 0.5, f0 = 0.25):
        """Generates a non-round Hermite function of a given order. The spectrogram of
        Hermite functions are given by an elipsoidal ridge in the time frequency plane,
        the center of which is given by (t0,f0).

        Args:
            order (int, optional): Order of the Hermite function. Defaults to 18.
            t0 (float, optional): Time coordinate of the center of the spectrogram.
            Defaults to 0.5.
            f0 (float, optional): Frequency coordinate of the center of the spectrogram.
            Defaults to 0.25.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """
        
        N = self.N
        t0 = int(N*t0)
        t = np.arange(N)-t0
        return hermite_fun(N, order, t=t, T = 1.5*np.sqrt(2*N))*np.cos(2*pi*f0*t)

    def signal_mc_triple_impulse(self, Nimpulses = 3):
        """Generates three equispaced impulses in time.

        Args:
            Nimpulses (int, optional): Number of impulses. Defaults to 3.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """
        
        N = self.N
        tmin = self.tmin
        tmax = N-tmin
        Nsub = tmax-tmin
        dloc = Nsub/(Nimpulses+1)
        impulses = np.zeros((Nsub,))
        signal = np.zeros((N,))
        for i in range(Nimpulses):
            impulses[int((i+1)*dloc -1 )] = 10
        
        signal[tmin:tmax] = impulses


        stft, stft_padded, Npad = get_stft(signal)
        for i in range(stft.shape[1]):
            stft[:,i] *= sg.windows.tukey(stft.shape[0],0.95)

        xr, t = reconstruct_signal_2(np.ones(stft.shape), stft_padded, Npad)
        return xr

    def signal_mc_impulses(self, Nimpulses = 5):
        """Generates equispaced impulses in time.

        Args:
            Nimpulses (int, optional): Number of impulses. Defaults to 3.

        Returns:
            numpy.ndarray: Returns a numpy array with the signal.
        """
        
        N = self.N
        tmin = self.tmin
        tmax = N-tmin
        Nsub = tmax-tmin
        dloc = Nsub/(Nimpulses+1)
        impulses = np.zeros((Nsub,))
        signal = np.zeros((N,))
        for i in range(Nimpulses):
            impulses[int((i+1)*dloc -1 )] = 2*(i+1)
        
        signal[tmin:tmax] = impulses

        stft, stft_padded, Npad = get_stft(signal)
        for i in range(stft.shape[1]):
            stft[:,i] *= sg.windows.tukey(stft.shape[0],0.95)

        xr, t = reconstruct_signal_2(np.ones(stft.shape), stft_padded, Npad)
        return xr
           
    

    

# def signal_mc_exp_chirps(self):
    #     """Generates a multicomponent signal comprising three exponential chirps.


    #     Returns:
    #         numpy.ndarray: Returns a numpy array with the signal.
    #     """
        
    #     N = self.N
    #     signal = np.zeros((N,))
    #     aux = np.zeros((N,))
    #     exponents = [4, 3, 2]
    #     finits = [self.fmin, 1.8*self.fmin, 2.5*self.fmin]
    #     fends = [0.3, 0.8, 1.2]
    #     ncomps = len(fends)

    #     max_freq = self.fmax

    #     for i in range(ncomps):
    #         _, instf, tmin, tmax = self.signal_exp_chirp(finit=finits[i],
    #                                                      fend=fends[i],
    #                                                      exponent=exponents[i],
    #                                                      r_instf=True)    

    #         instf2 = instf            
    #         if instf[0] >= max_freq:
    #             break
            
    #         instf2 = instf2[np.where(instf2 < max_freq)]
    #         tukwin = sg.windows.tukey(len(instf2),0.25)

    #         self.check_inst_freq(instf2)
    #         phase = np.cumsum(instf2)
    #         x = np.cos(2*pi*phase)
    #         tukwin = sg.windows.tukey(len(x),0.25)
    #         x = x*tukwin
    #         signal[tmin:tmin+len(x)] = x

    #         aux += signal
    #     return aux

    # def signal_mc_modulated_tones(self):
    #     return self.signal_mc_multi_cos()


    # def signal_mc_modulated_tones_2(self):
    #     return self.signal_mc_multi_cos_2()

    

    
    
