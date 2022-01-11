import numpy as np
from benchmark_demo.SignalBank import SignalBank
import pandas as pd
import numbers

# import json
# import pickle
# import sys
# import warnings


class Benchmark:
    """
    This class performs a number of tasks for methods comparison.
    """

    def __init__(self, task = 'denoising', methods = None, N = 256, parameters = None, SNRin = None, repetitions = None, checks = True):
        """
        This constructor parse the inputs and instantiate the object attributes
        """
        # Objects attributes
        self.task = None
        self.methods = None
        self.N = None
        self.repetitions = None
        self.SNRin = None
        self.results = None

        # Check input parameters and initialize the object attributes
        self.inputParsing(task, methods, N, parameters, SNRin, repetitions)

        # Generates a dictionary of signals
        self.bank = SignalBank(N)

        # Set the performance function according to the selected task
        self.comparisonFunction = self.setComparisonFunction(task)
        

    def inputParsing(self, task, methods, N, parameters, SNRin, repetitions):
        """
        Check input parameters and initialize the object attributes
        """

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

        # If no parameters are given to test.
        if parameters is None:
            self.parameters = {key: None for key in methods.keys}
        else:
            if type(methods) is dict:
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


    def methodsOutputCheck(self,output,input):
        if self.task == 'denoising':
            if type(output) is not np.ndarray:
                raise ValueError("Method's output should be a numpy array for task='denoising'.\n")

            if output.shape != input.shape:
                raise ValueError("Method's output should have the same shape as input for task='denoising'.\n")


    def setComparisonFunction(self, task):
        """
        Define different comparison functions for each task.
        """
        compFuncs = {
            'denoising': snrComparison,
            'detection': lambda base_signal, method_output: method_output 
        }
        return compFuncs[task]    


    def runTest(self, verbosity = False):
        """
        Run benchmark with the set parameters.
        """
        print('Running benchmark...')

        # Dictionaries for the results. This helps to express the results later using a DataFrame.
        results_dic = dict()
        params_dic = dict()
        method_dic = dict()
        SNR_dic = dict()

        # These loops run all the experiments and save the results in nested dictionaries.
        for signal_id in self.bank.signalDict:
            base_signal = self.bank.signalDict[signal_id]()
            for SNR in self.SNRin:
                noisy_signals, noise = add_snr_block(base_signal,SNR,self.repetitions)
                for method in self.methods:                        
                    for p,params in enumerate(self.parameters[method]):    
                        method_output = self.methods[method](noisy_signals,params)
                        self.methodsOutputCheck(method_output,noisy_signals) # Just checking if the output its valid.   

                        result =  self.comparisonFunction(base_signal, method_output)             
                        params_dic['Params'+str(p)] = result
                    
                    method_dic[method] = params_dic    
                    params_dic = dict()
           
                SNR_dic[SNR] = method_dic
                method_dic = dict()

            results_dic[signal_id] = SNR_dic   
            SNR_dic = dict()
        
        # Save dictionary in disc to use later.
        # json.dump( method_dic, open( "results_benchmark.json", 'w' ) )  
        # with open('saved_dictionary.pkl', 'wb') as f:
        #     pickle.dump(results_dic, f)


        self.results = results_dic # Save results for later.
        print('The test has finished.')
        return results_dic


    def getResults(self, results = None):
        if results is None:
            df = dic2df(self.results)
        else:
            df = dic2df(self.results)

        df = pd.concat({param: df[param] for param in df.columns})
        df = df.unstack(level=2)
        df = df.reset_index()
        df.columns = pd.Index(['Parameter','Signal_id', 'Method', 'Repetition'] + self.SNRin)
        df=df.reindex(columns=['Method', 'Parameter','Signal_id', 'Repetition'] + self.SNRin)
        df = df.sort_values(by=['Method', 'Parameter','Signal_id'])
        return df

# Pendientes: Guardar resultados, guardar senales y el ruido, guardar parametros en archivo. Verbosity.


# Other functions:
def dic2df(midic):
    """
    This function transforms a dictionary of arbitrary depth into a pandas' DataFrame object.
    """
    auxdic = dict()
    for key in midic:
        if isinstance(midic[key], dict):
            df = dic2df(midic[key])
            auxdic[key] = df       
        else:
            return pd.DataFrame(midic)
    
    # print(auxdic)
    df = pd.concat(auxdic,axis = 0)
    # print(df)
    return df


def add_snr_block(x,snr,K = 1):
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


def snrComparison(x,x_hat):
    """
    Quality reconstruction factor for denoising performance characterization.
    """
    qrf = np.zeros((x_hat.shape[0],))
    for i in range(x_hat.shape[0]):
        qrf[i] = 10*np.log10(np.sum(x**2)/np.sum((x_hat[i,:]-x)**2))
    
    return qrf