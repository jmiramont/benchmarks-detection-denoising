import warnings
import numpy as np
from benchmark_demo.SignalBank import SignalBank
import pandas as pd
import json
import pickle

class Benchmark:
    """This class performs a number of tasks for methods comparison."""

    def __init__(self, task = 'denoising', methods = None, N = None, parameters = None, SNRin = None, repetitions = None):
        self.methods = methods
        self.repetitions = repetitions
        # self.parameters = {i: param for i,param in enumerate(parameters)}
        # self.parameters = {'Param'+str(i): param for i,param in enumerate(parameters)}
        self.parameters = parameters
        self.bank = SignalBank(N)
        self.SNRin = SNRin
        self.comparisonFunction = self.setComparisonFunction(task)
        self.results = None
        
        if not self.inputsControl():
            print('Something went wrong, check input paramaters!')

    def inputsControl(self):
        # Check both dictionaries have the same keys:
        if not (self.methods.keys() == self.parameters.keys()):
            print('Both methods and parameters dictionaries should have the same keys.')
            return False    

        return True

    def setComparisonFunction(self, task):
        """Define different comparison functions for each task."""
        compFuncs = {
            'denoising': snrComparison,
            'detection': lambda base_signal, method_output: method_output 
        }
        return compFuncs[task]    


    def runTest(self, verbosity = False):
        """ Run benchmark with the set parameters."""
        print('Running benchmark...')
        results_dic = dict()
        params_dic = dict()
        method_dic = dict()
        SNR_dic = dict()

        
        for signal_id in self.bank.signalDict:
            base_signal = self.bank.signalDict[signal_id]()
            for SNR in self.SNRin:
                noisy_signals, noise = add_snr_block(base_signal,SNR,self.repetitions)
                for method in self.methods:                        
                    for p,params in enumerate(self.parameters[method]):    
                        method_output = self.methods[method](noisy_signals,params)
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
        with open('saved_dictionary.pkl', 'wb') as f:
            pickle.dump(results_dic, f)


        self.results = results_dic # Save results for later.

        print('The test has finished.')
        return results_dic


    def getResults(self):
        df = dic2df(self.results)
        df = pd.concat({param: df[param] for param in df.columns})
        df = df.unstack(level=2)
        df = df.reset_index()
        df.columns = pd.Index(['Parameter','Signal_id', 'Method', 'Repetition'] + self.SNRin)
        df=df.reindex(columns=['Method', 'Parameter','Signal_id', 'Repetition'] + self.SNRin)
        df = df.sort_values(by=['Method', 'Parameter','Signal_id'])
        return df

# Pendientes: Guardar resultados, guardar senales y el ruido, guardar parametros en archivo. Verbosity.

def dic2df(midic):
    """Transforms a dictionary of arbitrary depth into a pandas' DataFrame object."""
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
    """ Quality reconstruction factor for denoising performance characterization.
    """
    
    qrf = np.zeros((x_hat.shape[0],))
    for i in range(x_hat.shape[0]):
        qrf[i] = 10*np.log10(np.sum(x**2)/np.sum((x_hat[i,:]-x)**2))
    
    return qrf