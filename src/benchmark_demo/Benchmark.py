import numpy as np
from benchmark_demo.SignalBank import SignalBank, Signal
import pandas as pd
import numbers
import pickle
import multiprocessing
import copy
import time

# from typing import Callable, is_typeddict
# import json
# import sys
# import warnings


class Benchmark:
    """
    This class performs a number of tasks for methods comparison.

    Methods
    -------
    input_parsing(task, methods, N, parameters, SNRin, repetitions, using_signals, verbosity, parallelize):
        Parse input parameters of the constructor of class Benchmark.
    
    check_methods_output(output,input):
        Check that the outputs of the method to benchmark fulfill the required type and shape.
    
    set_comparison_function(task):
        Set the performance function for the selected task (future tasks could easily add new performance functions)
    
    inner_loop(benchmark_parameters):
        Main loop of the Benchmark.

    run_test(self):
        Run the benchmark.

    save_to_file(self,filename = None):
        Save the results to a binary file that encodes the benchmark object.
        Notice that the methods associated with the benchmark, not being pickable objects,
        are NOT saved.

    get_results_as_df(self, results = None):
        Get a pandas DataFrame object with the results of the benchmark.
    
    """

    def __init__(self,
                 task='denoising',
                 methods=None,
                 N=256, 
                 Nsub = None, 
                 parameters=None,
                 SNRin=None, 
                 repetitions=None, 
                 using_signals='all', 
                 verbosity=1,
                 parallelize = False,
                 complex_noise = False,
                 obj_fun = None):
        """ Initialize the main parameters of the test bench before running the benchmark.

        Args:
            task (str, optional): The task to test the methods. Defaults to 'denoising'.
            
            methods (dict, optional): A dictionary of functions. Defaults to None.
            
            N (int, optional): Lengths of the observation window for the signal
            generation. Defaults to 256.

            Nsub (int, optional): Lengths of the signal. If None, uses Nsub (See
            SignalBank class description for details). Defaults to None.
            
            parameters (dict, optional): A dictionary of parameters for the methods
            to run. The keys of this dictionary must be same as the methods dictionary. 
            Defaults to None.
            
            SNRin (tuple, optional): List or tuple with the SNR values. 
            Defaults to None.
            
            repetitions (int, optional): Number of times each method is applied for 
            each value of SNR. This value is the number of noise realizations that are 
            used to assess the methods. Defaults to None.
            
            using_signals (tuple, optional): Tuple or list of the signal ids from the 
            SignalBank class. Defaults to 'all'.
            
            verbosity (int, optional): Number from 0 to 4. It determines the number of 
            messages passed to the console informing the progress of the benchmarking 
            process. Defaults to 1.
            
            parallelize (bool, optional): If True, tries to run the process in parallel.
            Defaults to False.

            complex_noise (bool, optional): If True, uses complex noise.
            Defaults to False.

            obj_fun (callable, optional): If None, used the default objective functions
            for benchmarking. If a function is passed as an argument, the default is
            overridden. 

        """

        # Objects attributes
        self.task = None
        self.methods = None
        self.N = None
        self.Nsub = None
        self.repetitions = None
        self.SNRin = None
        self.results = None
        self.verbosity = None
        self.complex_noise = None
        self.noise_matrix = None
        self.methods_and_params_dic = dict()
        
        # Check input parameters and initialize the object attributes
        self.input_parsing(task,
                           methods,
                           N, 
                           Nsub, 
                           parameters, 
                           SNRin, 
                           repetitions, 
                           using_signals, 
                           verbosity, 
                           parallelize,
                           complex_noise,
                           obj_fun)

        # Parallelize parameters
        if self.parallel_flag:
            if self.verbosity > 1:
                print("Number of processors: ", multiprocessing.cpu_count())    
                print('Parallel pool: {}'.format(self.processes))


        # Generates a dictionary of signals
        signal_bank = SignalBank(N, Nsub=self.Nsub, return_signal=True)
        self.tmin = signal_bank.tmin # Save initial and end times of signals.
        self.tmax = signal_bank.tmax
        if Nsub is None:
            self.Nsub = signal_bank.Nsub

        # print(self.tmin,self.tmax,self.Nsub)    

        self.signal_dic = signal_bank.signalDict
        if using_signals == 'all':
            self.signal_ids = [akey for akey in self.signal_dic]
        else:
            self.signal_ids = using_signals




    def input_parsing(self, 
                    task, 
                    methods, 
                    N,
                    Nsub, 
                    parameters, 
                    SNRin, 
                    repetitions, 
                    using_signals, 
                    verbosity, 
                    parallelize,
                    complex_noise,
                    obj_fun):

        """Parse input parameters of the constructor of class Benchmark.

        Args:
            task (str, optional): The task to test the methods. Defaults to 'denoising'.
            methods (dict, optional): A dictionary of functions. Defaults to None.
            N (int, optional): Lengths of the signals. Defaults to 256.
            parameters (dict, optional): A dictionary of parameters for the methods
            to run. The keys of this dictionary must be same as the methods dictionary. 
            Defaults to None.
            SNRin (tuple, optional): List or tuple with the SNR values. 
            Defaults to None.
            repetitions (int, optional): Number of times each method is applied for 
            each value of SNR.
            This value is the number of noise realizations that are used to assess the 
            methods.Defaults to None.
            using_signals (tuple, optional): Tuple or list of the signal ids from the 
            SignalBank class. Defaults to 'all'.
            verbosity (int, optional): Number from 0 to 4. It determines the number of 
            messages passed to the console informing the progress of the benchmarking 
            process. Defaults to 1.
            parallelize (bool, optional): If True, tries to run the process in parallel.
            Defaults to False.
        
        Raises:
            ValueError: If any parameter is not correctly parsed.
        """
        # Check verbosity
        assert isinstance(verbosity,int) and 0<=verbosity<6 , 'Verbosity should be an integer between 0 and 5'
        self.verbosity = verbosity

        # TODO: ADD NEW TASKS    
        # Check the task is either 'denoising' or 'detection'.
        # if (task != 'denoising' and task != 'detection'):
        #     raise ValueError("The tasks should be either 'denoising' or 'detection'.\n")
        # else:
        self.task = task

        # Check methods is a dictionary
        if type(methods) is not dict:
            raise ValueError("Methods should be a dictionary.\n")
        else:
            self.methods = methods
            self.methods_ids = [llave for llave in methods]

        # If no parameters are given to the benchmark.
        if parameters is None:
            self.parameters = {key: (((),{}),) for key in methods.keys()}
        else:
            if type(parameters) is dict:
                self.parameters = parameters
            else:
                raise ValueError("Parameters should be a dictionary or None.\n")

        #Check both dictionaries have the same keys:
        if not (self.methods.keys() == self.parameters.keys()):
            # sys.stderr.write
            raise ValueError("Both methods and parameters dictionaries should have the same keys.\n")

        # If we are here, this is a new benchmark, so the all the methods are new:
        self.this_method_is_new = {method:True for method in self.methods_ids}

        # Check if N is an entire:
        if type(N) is int:
            self.N = N
        else:
            raise ValueError("N should be an entire.\n")

        # Check if Nsub is an entire:
        if Nsub is not None:
            if type(Nsub) is not int:
                raise ValueError("Nsub should be an entire.\n")

            # Check if Nsub is lower than N:
            if self.N > Nsub:
                self.Nsub = Nsub
            else:
                raise ValueError("Nsub should be lower than N.\n")

        # Check if SNRin is a tuple or list, and if so, check if there are only numerical variables.
        if (type(SNRin) is tuple) or (type(SNRin) is list):
            for i in SNRin:
                if not isinstance(i, numbers.Number):
                    raise ValueError("All elements in SNRin should be real numbers.\n")

            self.SNRin = SNRin
        else:
            raise ValueError("SNRin should be a tuple or a list.\n")

        # Check if repetitions is an entire:
        if type(repetitions) is int:
            self.repetitions = repetitions
        else:
            raise ValueError("Repetitions should be an entire.\n")

        # Check what to do with list of signals:
        if using_signals !='all':
            if isinstance(using_signals,tuple) or isinstance(using_signals,list):
                signal_bank = SignalBank(N)
                llaves = signal_bank.signalDict.keys()
                assert all(signal_id in llaves for signal_id in using_signals)

        # Check if complex_noise flag is bool:
        if type(complex_noise) is bool:
            self.complex_noise = complex_noise
        else:
            raise ValueError("'complex_noise' should be a bool.\n")
            
        # Handle parallelization parameters:
        max_processes = multiprocessing.cpu_count()

        if parallelize is False:
            self.parallel_flag = False
        else:
            if parallelize is True:
                    if parallelize:   
                        available_proc = multiprocessing.cpu_count()    
                        self.processes = np.max((1, available_proc//2 ))
                        self.parallel_flag = True
            else:    
                if isinstance(parallelize,int):
                    if parallelize < max_processes:
                        self.processes = parallelize
                    else:
                        self.processes = max_processes
                    self.parallel_flag = True

        # Check objective function
        # Set the default performance function according to the selected task
        if obj_fun is None:
            self.objectiveFunction = self.set_objective_function(task)
        else:
            if callable(obj_fun):
                self.objectiveFunction = obj_fun
            else:
                raise ValueError("'obj_fun' should be a callable object.\n")
                

    def check_methods_output(self, output, input):
        """Check that the outputs of the method to benchmark fulfill the required type 
        and shape.

        Args:
            output: Output from the method. The type and shape depends on the task.
            input: Input passed to the method to produce output.

        Raises:
            ValueError: If the output does not comply with the required type and shape 
            for the selected task.
        """
        if self.task == 'denoising':
            if type(output) is not np.ndarray:
                raise ValueError("Method's output should be a numpy array for task='denoising'.\n")

            if output.shape != input.shape:
                raise ValueError("Method's output should have the same shape as input for task='denoising'.\n")


    def set_objective_function(self, task):
        """
        Set the performance function for the selected task (future tasks could easily add new performance functions)
        """
        
        compFuncs = {#'denoising': lambda x: self.snr_comparison(*x,tmin=self.tmin,tmax=self.tmax),
                    'denoising': self.snr_comparison,
                    'detection': detection_perf_function,
                    'component_denoising':compare_qrf_block,
                    'inst_frequency':compare_instf_block,
                    }
        return compFuncs[task]    


    def inner_loop(self, benchmark_parameters,timer=False):
        """Main loop of the Benchmark.

        Args:
            benchmark_parameters (tuple): Tuple or list with the parameters of the benchmark.
            timer (bool): If true, measures the time of execution.

        Returns:
            narray: Return a numpy array, the shape of which depends on the selected task.
        """
        
        method, params, idx = benchmark_parameters

        if self.verbosity >= 5:
            print('------ Inner loop. ' + method + ': ' + str(idx), flush=True )

        # Get the noisy signal (as a ndarray) and wrap it with the Signal class, adding
        # the signal information from the base signal.
        # This wrapper class behaves like a numpy array, but encapsulates signal info, 
        # like the total number of components or number of components per time.

        noisy_signal = Signal(self.noisy_signals[idx])
        noisy_signal.ncomps = self.base_signal_info['ncomps']
        noisy_signal.total_comps = self.base_signal_info['total_comps']


        try:
            args, kwargs = params
            tstart = time.time()   
            method_output = self.methods[method](noisy_signal,*args,**kwargs)
            elapsed = time.time() - tstart

        except BaseException as err:
            print(f"Unexpected error {err=}, {type(err)=} in method {method}. Watch out for NaN values.")
            
            # TODO Write a log here with the error. Should identify the repetition.
            
            if self.task == 'denoising':
                method_output = np.empty(noisy_signal.shape)
                method_output[:] = np.nan

            if self.task == 'detection':
                method_output = np.nan

        #! Rewrite this part.
        # self.check_methods_output(method_output,noisy_signals) # Just checking if the output its valid.   
        
        return method_output, elapsed

    def run_test(self):
        """Run the benchmark.

        Returns:
            dict: Returns nested dictionaries with the results of the benchmark.
        """
        if self.verbosity > 0:
            print('Running benchmark...')

        # Dictionaries for the results. This helps to express the results later using a DataFrame.
        if self.results is None:
            self.results = dict()
            self.elapsed_time = dict()

            # Create dictionary tree:
            for signal_id in self.signal_ids: 
                SNR_dic = dict()
                for SNR in self.SNRin:
                    method_dic = dict()
                    for method in self.methods:
                        params_dic = dict()
                        # for params in self.parameters[method]:
                        #     params_dic[str(params)] = 'Deep' 
                        method_dic[method] = params_dic
                        self.elapsed_time[method]  = params_dic                     

                    SNR_dic[SNR] = method_dic
                self.results[signal_id] = SNR_dic        
        
        

        # This run all the experiments and save the results in nested dictionaries.
        for signal_id in self.signal_ids:
            if self.verbosity >= 1:
                print('- Signal '+ signal_id)

            self.base_signal = self.signal_dic[signal_id]()
            self.base_signal_info = self.signal_dic[signal_id]().get_info()
            
            for SNR in self.SNRin:
                if self.verbosity >= 2:
                    print('-- SNR: {} dB'.format(SNR))

                # If the benchmark has been run before, re-run again with the same noise.
                if self.noise_matrix is None:
                    self.noise_matrix = self.generate_noise()
                            
                noisy_signals = self.sigmerge(self.base_signal, 
                                            self.noise_matrix,
                                            SNR,
                                            tmin = self.tmin,
                                            tmax = self.tmax)

                # Access current noisy signals from the main loop.                                                    
                self.noisy_signals = noisy_signals

                
                #===========================MAIN LOOP===================================

                # ------------------------- Parallel loop ------------------------------
                if self.parallel_flag:
                    parallel_list = list()
                    for method in self.methods:
                        if self.verbosity >= 1:
                            print('--- Parallel loop -- Method: '
                                +method
                                +'(all parameters)')                                               
                        for p,params in enumerate(self.parameters[method]):
                            args, kwargs = get_args_and_kwargs(params)
                            for idx, noisy_signal in enumerate(noisy_signals):
                                parallel_list.append([method, (args, kwargs), idx])

                    # Here implement the parallel stuff
                    pool = multiprocessing.Pool(processes=self.processes) 
                    parallel_results = pool.map(self.inner_loop, parallel_list) 
                    pool.close() 
                    pool.join()
                    if self.verbosity >= 1:    
                        print('--- Parallel loop finished.') 

                # ---------------------- Serial loop -----------------------------------
                k = 0  # This is used to get the parallel results if it's necessary.
                
                for method in self.methods:
                    
                    if self.this_method_is_new[method]:
                        params_dic = dict()
                        if self.verbosity >= 3:
                            print('--- Method: '+ method)                    

                        for p, params in enumerate(self.parameters[method]):
                            elapsed = []

                            if self.verbosity >= 4:
                                print('---- Parameters Combination: '+ str(p)) 
                            
                            args, kwargs = get_args_and_kwargs(params)
                            
                            if self.task == 'component_denoising':
                                extrargs = {'tmin':self.tmin,'tmax':self.tmax}
                                method_output = [[] for aaa in range(noisy_signals.shape[0])]

                            if self.task == 'inst_frequency':
                                extrargs = {'tmin':self.tmin,'tmax':self.tmax}
                                method_output = [[] for aaa in range(noisy_signals.shape[0])]                                
                            
                            if self.task == 'denoising':
                                extrargs = {'tmin':self.tmin,'tmax':self.tmax}
                                method_output = np.zeros_like(noisy_signals)

                            if self.task == 'detection':
                                extrargs = {}
                                method_output = np.zeros((self.repetitions)).astype(bool)
                                                         
                            for idx, noisy_signal in enumerate(noisy_signals):
                                if self.parallel_flag:  # Get results from parallel...
                                    tmp, extime = parallel_results[k]
                                    method_output[idx] = tmp
                                    # Save but DON'T TRUST the exec. time in parallel.
                                    elapsed.append(extime) 
                                    k += 1     
                                else:                   # Or from serial computation.
                                    
                                    tmp, extime = self.inner_loop([method,
                                                        (args, kwargs), 
                                                        idx])        
                                    method_output[idx] = tmp
                                    elapsed.append(extime)
                                    
                            # Either way, results are saved in a nested dictionary.
                            result =  self.objectiveFunction(self.base_signal, 
                                                            method_output,
                                                            **extrargs)
                        
                            # params_dic['Params'+str(p)] = result
                            params_dic[str(params)] = result                        
                            self.elapsed_time[method][str(params)] = elapsed

                            if self.verbosity > 4:
                                print('Elapsed:{}'.format(np.mean(elapsed)))                    

                        self.results[signal_id][SNR][method] = params_dic
                        self.methods_and_params_dic[method] = [key for key in params_dic] 

            #   method_dic[method] = params_dic    
                        
           
            #     SNR_dic[SNR] = method_dic
            #     method_dic = dict()

            # self.results[signal_id] = SNR_dic   
            # SNR_dic = dict()

        # self.results = results_dic # Save results for later.
        if self.verbosity > 0:
            print('The test has finished.')
        
        # Don't use old methods if run again.
        for method in self.this_method_is_new:
            self.this_method_is_new[method] = False

        return self.results

        
    def save_to_file(self,filename = None):
        """Save the results to a binary file that encodes the benchmark object.
        Notice that the methods associated with the benchmark, not being pickable objects,
        are NOT saved.

        Args:
            filename (str, optional): Path and filename. Defaults to None.

        Returns:
            bool: True if the file was successfully created.
        """

        if filename is None:
            filename = 'a_benchmark'

        a_copy = self
        a_copy.methods = {key:None for key in a_copy.methods}
        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(a_copy, f)    

        return True


    def get_results_as_df(self, results = None):
        """Get a pandas DataFrame object with the results of the benchmark.

        Args:
            results (dict, optional): Nested dictionary with the results of the 
            benchmark. Defaults to None.

        Returns:
            DataFrame: Returns a pandas DataFrame with the results.
        """
        if results is None:
            df = self.dic2df(self.results)
        else:
            df = self.dic2df(self.results)

        df = pd.concat({param: df[param] for param in df.columns})
        df = df.unstack(level=2)
        df = df.reset_index()
        df.columns = pd.Index(['Parameter','Signal_id', 'Method', 'Repetition'] + self.SNRin)
        df=df.reindex(columns=['Method', 'Parameter','Signal_id', 'Repetition'] + self.SNRin)
        df = df.sort_values(by=['Method', 'Parameter','Signal_id'])

        aux2 = np.zeros((df.shape[0],))
        for metodo in self.methods_and_params_dic:
            aux = np.zeros((df.shape[0],))
            for params in self.methods_and_params_dic[metodo]:
                aux = aux | ( (df['Parameter'] == params)&(df['Method']==metodo) )
            aux2 = aux2 | aux
        
        df2 = df[aux2]
        return df2


    def add_new_method(self, methods, parameters=None):
        # Check methods is a dictionary and update existing dictionary of methods.
        if type(methods) is not dict:
            raise ValueError("Methods should be a dictionary.\n")
            
        # If no parameters are given to the benchmark.
        if parameters is None:
            parameters = {key: (((),{}),) for key in methods}
        else:
            if type(parameters) is not dict:
                raise ValueError("Parameters should be a dictionary or None.\n")

        for key in methods:
            if key not in self.methods.keys():
                self.methods[key] = methods[key]
                self.methods_ids.append(key)
                self.parameters[key] = parameters[key]
                self.elapsed_time[key]  = dict() 
                self.this_method_is_new[key] = True

        #Check both dictionaries have the same keys:
        if not (self.methods.keys() == self.parameters.keys()):
            # sys.stderr.write
            raise ValueError("Both methods and parameters dictionaries should have the same keys.\n")

        # New methods cannot be parallelized (for now).
        self.parallel_flag = False


# Other functions:
    def dic2df(self, mydic):
        """_summary_

        Args:
            mydic (_type_): _description_

        Returns:
            _type_: _description_
        """
        auxdic = dict()
        for key in mydic:
            if isinstance(mydic[key], dict):
                df = self.dic2df(mydic[key])
                auxdic[key] = df       
            else:
                return pd.DataFrame(mydic)
        
        df = pd.concat(auxdic,axis = 0)
        return df

    def generate_noise(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        noise_matrix = np.random.randn(self.repetitions,self.N)
        if self.complex_noise:
            noise_matrix += 1j*np.random.randn(self.repetitions,self.N)

        return noise_matrix


    # Static methods:
    @staticmethod
    def sigmerge(x1, noise, ratio, tmin=None, tmax=None, return_noise=False):
        # Get signal parameters.
        N = len(x1)

        if tmin is None:
            tmin=0
        if tmax is None:
            tmax=N

        sig = np.random.randn(*noise.shape)
        ex1=np.mean(np.abs(x1[tmin:tmax])**2)

        if len(noise.shape)==1:
            ex2=np.mean(np.abs(noise)**2)
        else:
            ex2=np.mean(np.abs(noise)**2, axis=1)
        h=np.sqrt(ex1/(ex2*10**(ratio/10)))

        if len(noise.shape)>1:
            h.resize((noise.shape[0],1))
            
        scaled_noise = noise*h
        # sig = sig*h.reshape((noise.shape[0],1))
        sig=x1+scaled_noise

        if return_noise:
            return sig, scaled_noise
        else:
            return sig
    
    @staticmethod
    def snr_comparison(x, x_hat, tmin=None, tmax=None):
        """
        Quality reconstruction factor for denoising performance characterization.
        """
        if tmin is None:
            tmin = 0

        if tmax is None:
            tmax = len(x)

        x = x[tmin:tmax]
        
        if len(x_hat.shape)==1:
            x_hat = x_hat[tmin:tmax]
            qrf = 10*np.log10(np.sum(x**2)/np.sum((x_hat-x)**2))
        else:
            x_hat = x_hat[:,tmin:tmax]
            qrf = np.zeros((x_hat.shape[0],))
            for i in range(x_hat.shape[0]):
                qrf[i] = 10*np.log10(np.sum(x**2)/np.sum((x_hat[i,:]-x)**2))
        
        return qrf


def detection_perf_function(original_signal, detection_output):
    return detection_output


def get_args_and_kwargs(params):
        if type(params) is dict:
                args = []
                kwargs = params
        else:
            dict_indicator = [type(i) is dict for i in params]
            if any(dict_indicator):
                assert len(params) == 2, "Parameters must be given as a dictionary or an iterable."
                for i in range(len(params)):
                    kwargs = params[np.where(dict_indicator)[0][0]]
                    args = params[np.where([not i for i in dict_indicator])[0][0]]
            else:
                args = params
                kwargs = dict()

        return args, kwargs


""" Performance metrics for new tasks

"""

from numpy import mean, abs 
from numpy.linalg import norm

def corr_comps(x, xest):
    idx = np.where(abs(x)>0)
    x_aux = x[idx]
    xest_aux = xest[idx]
    cor = (
        abs(sum((x_aux-mean(x_aux)) * (xest_aux-mean(xest_aux)))) 
        /(norm(x_aux-mean(x_aux)) * norm(xest_aux-mean(xest_aux))+ 1e-15)
            )
    return cor

def mse(x, xest):
    assert len(x) == len(xest), 'Should be of equal length.'
    idx = np.where(abs(x)>0)
    x_aux = x[idx]
    xest_aux = xest[idx]
    error = np.mean((x_aux-xest_aux)**2)
    return error

def compute_qrf(x, x_hat, tmin=None,tmax=None):
    """
    Quality reconstruction factor
    """
    if tmin is None:
        tmin = 0
    
    if tmax is None:
        tmax = len(x)

    x = x[tmin:tmax]
    x_hat = x_hat[tmin:tmax]
    qrf = 10*np.log10(np.sum(x**2)/np.sum((x_hat-x)**2))
    return qrf


# def order_components(Xest, X, metric = corr_comps):
#     order = []
#     values = np.array([[metric(x,xest) for x in X] for xest in Xest])
#     # order = np.argmax(values, axis=0)
#     for i in range(values.shape[1]):
#         col = values[:,i]
#         sort_col = np.sort(col)[-1::-1]
#         for j in range(len(sort_col)):
#             row = values[np.where(col == sort_col[j])]
#             if sort_col[j] == np.max(row):
#                 order.append(np.where(col == sort_col[j])[0][0])
#                 break
#     return order

def order_components(Xest, X, minormax = 'max', metric = corr_comps):
    order = [[] for aaa in range(len(X))]
    values = np.array([[metric(x,xest) for x in X] for xest in Xest], dtype=object)
    if minormax=='max':
        fun = np.argmax
        factor = -1
    if minormax == 'min':
        fun = np.argmin
        factor = 1

    while np.any(np.array([k == [] for k in order], dtype=object)):
        ind = np.unravel_index(fun(values, axis=None), values.shape)
        if (ind[0] not in order) and (order[ind[1]] == []):
            order[ind[1]] = ind[0]
        values[ind] = factor*np.inf
    return order    


def compare_qrf_block(signal, method_output, tmin=None, tmax=None):
    X = signal.comps
    output = []
    for Xest in method_output:
        order = order_components(Xest, X)
        Xaux = Xest[order]
        qrfs = []
        for x,xaux in zip(X,Xaux):
            indx = np.where(np.abs(x)>0)
            qrfs.append(compute_qrf(x[indx], xaux[indx],tmin=tmin,tmax=tmax))
        output.append(qrfs)    
    output = np.array(output, dtype=object)
    dict_output = {'Comp.{}'.format(i):output[:,i] for i in range(output.shape[1])}
    return dict_output

    
def compare_instf_block(signal, method_output, tmin=None, tmax=None):
    X = signal.instf
    output = []
    for Xest in method_output:
        order = order_components(Xest, X, minormax = 'min', metric = mse)
        Xaux = Xest[order]
        qrfs = []
        for x,xaux in zip(X,Xaux):
            indx = np.where(np.abs(x)>0)
            # qrfs.append(compute_qrf(x[indx], xaux[indx],tmin=tmin,tmax=tmax))
            qrfs.append(mse(x[indx], xaux[indx]))
        output.append(qrfs)    
    output = np.array(output, dtype=object)
    dict_output = {'Comp.{}'.format(i):output[:,i] for i in range(output.shape[1])}
    return dict_output    



# ! Deprecated
# Some neat solution I saw online for time measuring.
# class Timer(object):
#     def __init__(self, name=None):
#         self.name = name

#     def __enter__(self):
#         self.tstart = time.time()

#     def __exit__(self, type, value, traceback):
#         if self.name:
#             print('[%s]' % self.name,)
#         print('Elapsed: %s' % (time.time() - self.tstart))


# 
# #    def add_snr_block(self, x, snr, K=1, complex_noise=False):
#         """
#         Creates K realizations of the signal x with white Gaussian noise, with SNR 
#         equal to 'snr'. SNR is defined as SNR (dB) = 10 * log10(Ex/En), where Ex and En 
#         are the energy of the signal and that of the noise respectively.
#         """
        
#         # Get signal parameters.
#         N = self.N
#         tmin = self.tmin
#         tmax = self.tmax
#         Nsub = self.tmax-self.tmin

#         # Make sure signal's mean is zero.
#         x = x - np.mean(x)
#         Px = np.sum(x ** 2)
        
#         # Create the noise for signal with given SNR:
#         n = np.random.randn(Nsub,K)
#         if complex_noise:
#             n = n.astype(complex)
#             n += 1j*np.random.randn(Nsub,K)
#         Pn = np.sum(np.abs(n) ** 2, axis = 0) # Normalize to 1 the variance of noise.
#         n = n / np.sqrt(Pn)
#         Pn = Px * 10 ** (- snr / 10) # Give noise the prescribed variance.
#         n = n * np.sqrt(Pn)

#         # Complete the signal with noise outside the Nsub samples
#         aux = np.random.randn(N,K)
#         if complex_noise:
#             aux = aux.astype(complex)
#             aux += 1j*np.random.randn(N,K)
#         Paux = np.sum(np.abs(aux) ** 2, axis = 0)
#         aux = aux / np.sqrt(Paux)
#         Paux = Px * 10 ** (- snr / 10)
#         aux = aux * np.sqrt(Paux)
#         aux[tmin:tmax,:] = n
#         n = aux

#         # Checking:
#         nprobe=n[:,1].T
#         Ex = np.sum(x[tmin:tmax]**2)
#         Eprobe = np.sum((nprobe[tmin:tmax])**2)
#         # Ex = np.sum(x**2)
#         # Eprobe = np.sum((nprobe)**2)
#         # print("SNRout={}".format(10*np.log10(Ex/Eprobe)))
    
#         return x+n.T, n.T        