import numpy as np
from benchmark_demo.SignalBank import SignalBank
import pandas as pd
import numbers
import pickle
# import json

# import sys
# import warnings


class Benchmark:
    """
    This class performs a number of tasks for methods comparison.
    """

    def __init__(self, task='denoising', methods=None, N=256, parameters=None,
                SNRin=None, repetitions=None, using_signals='all', checks=True, verbosity=1,
                from_file = None):
        """
        This constructor parse the inputs and instantiate the object attributes
        """
        if from_file is not None:
            self = self.load_from_file(filename = from_file)

        else:
            # Objects attributes
            self.task = None
            self.methods = None
            self.N = None
            self.repetitions = None
            self.SNRin = None
            self.results = None
            self.verbosity = None
            self.methods_and_params_dic = dict()
            # Check input parameters and initialize the object attributes
            if checks:
                self.input_parsing(task, methods, N, parameters, SNRin, repetitions, using_signals, verbosity)

            # Generates a dictionary of signals
            signal_bank = SignalBank(N)
            self.signal_dic = signal_bank.signalDict

            if using_signals == 'all':
                self.signal_ids = [llave for llave in self.signal_dic]
            else:
                self.signal_ids = using_signals


            # Set the performance function according to the selected task
            self.comparisonFunction = self.set_comparison_function(task)
        

    def input_parsing(self, task, methods, N, parameters, SNRin, repetitions, using_signals, verbosity):
        """
        Check input parameters and initialize the object attributes
        """
        # Check verbosity
        assert isinstance(verbosity,int) and 0<=verbosity<5 , 'Verbosity should be an integer between 0 and 4'
        self.verbosity = verbosity

        # Check the task is either 'denoising' or 'detecting'
        if (task != 'denoising' and task != 'detecting'):
            raise ValueError("The tasks should be either 'denoising' or 'detecting'.\n")
        else:
            self.task = task

        # Check methods is a dictionary
        if type(methods) is not dict:
            raise ValueError("Methods should be a dictionary.\n")
        else:
            self.methods = methods
            self.methods_ids = [llave for llave in methods]

        # If no parameters are given to the benchmark.
        if parameters is None:
            self.parameters = {key: [0,] for key in methods.keys()}
        else:
            if type(parameters) is dict:
                self.parameters = parameters
            else:
                raise ValueError("Parameters should be a dictionary or None.\n")

        #Check both dictionaries have the same keys:
        if not (self.methods.keys() == self.parameters.keys()):
            # sys.stderr.write
            raise ValueError("Both methods and parameters dictionaries should have the same keys.\n")

        # Check if N is an entire:
        if type(N) is int:
            self.N = N
        else:
            raise ValueError("N should be an entire.\n")

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


    def check_methods_output(self,output,input):
        if self.task == 'denoising':
            if type(output) is not np.ndarray:
                raise ValueError("Method's output should be a numpy array for task='denoising'.\n")

            if output.shape != input.shape:
                raise ValueError("Method's output should have the same shape as input for task='denoising'.\n")


    def set_comparison_function(self, task):
        """
        Define different comparison functions for each task.
        """
        compFuncs = {
            'denoising': snr_comparison,
            'detection': lambda base_signal, method_output: method_output 
        }
        return compFuncs[task]    


    def run_test(self):
        """
        Run benchmark with the set parameters.
        """
        if self.verbosity > 0:
            print('Running benchmark...')

        # Dictionaries for the results. This helps to express the results later using a DataFrame.
        results_dic = dict()
        params_dic = dict()
        method_dic = dict()
        SNR_dic = dict()

        # These loops here run all the experiments and save the results in nested dictionaries.
        for signal_id in self.signal_ids:
            if self.verbosity > 1:
                print('- Signal '+ signal_id)

            base_signal = self.signal_dic[signal_id]()
            for SNR in self.SNRin:
                if self.verbosity > 2:
                    print('-- SNR: {} dB'.format(SNR))

                noisy_signals, noise = self.add_snr_block(base_signal,SNR,self.repetitions)
                for method in self.methods:    
                    if self.verbosity > 3:
                        print('--- Method: '+ method)                    

                    for p,params in enumerate(self.parameters[method]):
                        try:    
                            method_output = self.methods[method](noisy_signals,params)
                        except BaseException as err:
                            print(f"Unexpected {err=}, {type(err)=}. Watch out for NaN values.")
                            method_output = np.empty(noisy_signals.shape)
                            method_output[:] = np.nan
                        self.check_methods_output(method_output,noisy_signals) # Just checking if the output its valid.   

                        result =  self.comparisonFunction(base_signal, method_output)             
                        params_dic['Params'+str(p)] = result
                    
                    self.methods_and_params_dic[method] = [llave for llave in params_dic]
                    method_dic[method] = params_dic    
                    params_dic = dict()
           
                SNR_dic[SNR] = method_dic
                method_dic = dict()

            results_dic[signal_id] = SNR_dic   
            SNR_dic = dict()

        self.results = results_dic # Save results for later.
        if self.verbosity > 0:
            print('The test has finished.')

        return results_dic


        
    def save_to_file(self,filename = None):
        """ Save results to file with filename"""
        if filename is None:
            filename = 'a_benchmark'
        a_copy = self
        a_copy.methods = None
        # benchmark_dict = {  'task': self.task,
        #                     'methods': self.methods,
        #                     'N': self.N,
        #                     'repetitions': self.repetitions,
        #                     'SNRin': self.SNRin,
        #                     'results': self.results,
        #                     'verbosity': self.verbosity,
        #                     'results': self.results,
        #                     'signal_dic': self.signal_dic,
        #                     'signal_ids': self.signal_ids
        # }


        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(a_copy, f)    

        return True


    def load_from_file(self, filename = None):
        """ Save results to file with filename"""
        if filename is None:
            filename = 'a_benchmark'

        with open(filename + '.pkl', 'rb') as f:
            benchmark = pickle.load(f)

        # self = benchmark
        # self.task = benchmark_dict['task']
        # # self.methods = benchmark_dict['methods']
        # self.N = benchmark_dict['N']
        # self.repetitions = benchmark_dict['repetitions']
        # self.SNRin = benchmark_dict['SNRin']
        # self.results = benchmark_dict['results']
        # self.verbosity = benchmark_dict['verbosity']  
        # self.results = benchmark_dict['results']  
        # self.signal_dic = benchmark_dict['signal_dic']  
        # self.signal_ids = benchmark_dict['signal_ids']  
        
        return benchmark


    def get_results_as_df(self, results = None):
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


# Other functions:
    def dic2df(self, midic):
        """
        This function transforms a dictionary of arbitrary depth into a pandas' DataFrame object.
        """
        auxdic = dict()
        for key in midic:
            if isinstance(midic[key], dict):
                df = self.dic2df(midic[key])
                auxdic[key] = df       
            else:
                return pd.DataFrame(midic)
        
        # print(auxdic)
        df = pd.concat(auxdic,axis = 0)
        # print(df)
        return df


    def add_snr_block(self, x, snr, K=1, complex_noise=False):
        """
        Creates K realizations of the signal x with white Gaussian noise, with SNR equal to snr.
        SNR is defined as SNR (dB) = 10 * log10(Ex/En), where Ex and En are the energy of the signal
        and that of the noise respectively.
        """
        N = len(x)
        x = x - np.mean(x)
        Px = np.sum(x ** 2)
        # print(x)

        n = np.random.rand(N,K)
        n = n - np.mean(n,axis = 0)

        Pn = np.sum(n ** 2, axis = 0)
        n = n / np.sqrt(Pn)

        Pn = Px * 10 ** (- snr / 10)
        n = n * np.sqrt(Pn)

        return x+n.T, n.T


def snr_comparison(x,x_hat):
    """
    Quality reconstruction factor for denoising performance characterization.
    """
    qrf = np.zeros((x_hat.shape[0],))
    for i in range(x_hat.shape[0]):
        qrf[i] = 10*np.log10(np.sum(x**2)/np.sum((x_hat[i,:]-x)**2))
    
    return qrf