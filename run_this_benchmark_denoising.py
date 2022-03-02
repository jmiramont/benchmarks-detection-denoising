import importlib
from methods import *
import time
# Collects all the methods in the folder/ module "methods" and make a global list of them.
modules = dir()
modules = [mod_name for mod_name in modules if mod_name.startswith('method_')]
global list_of_methods # Use with caution.
list_of_methods = list()    
for mod_name in modules:
    mod = importlib.import_module('methods.' + mod_name)
    # print(mod)
    method = getattr(mod, 'instantiate_method')
    list_of_methods.append(method())

from benchmark_demo.Benchmark import *
import numpy as np
from benchmark_demo.ResultsInterpreter import ResultsInterpreter


dictionary_of_methods = dict()
dictionary_of_parameters = dict()

# Select only those methods for denoising.

for method_instance in list_of_methods:
    if method_instance.get_task() == 'denoising':
        method_id = method_instance.get_method_id()
        dictionary_of_methods[method_id] = method_instance.method
        dictionary_of_parameters[method_id] = method_instance.get_parameters()

SNRin = [0, 5]#, 10, 20, 30]

# Standard test:
signal_names = ['LinearChirp', 'CosChirp', 'McPureTones', # Single-component signals
                'McCrossingChirps',                       # Crossing-components  
                'McHarmonic','McPureTones',]               # Multi-Component Harmonic signals  
                # 'McModulatedTones','McDoubleCosChirp',    # Multi-Component Non-Harmonic  
                # 'McSyntheticMixture','McSyntheticMixture2',
                # 'HermiteFunction','HermiteElipse',        # Hermite type signals  
                # 'ToneDumped','ToneSharpAttack',           # Dumped and Sharps attacks  
                # 'McOnOffTones']                           # Modes that born and die

if __name__ == "__main__":
    my_benchmark = Benchmark(task = 'denoising',
                            methods = dictionary_of_methods,
                            N = 512,
                            parameters = dictionary_of_parameters, 
                            SNRin = SNRin,
                            using_signals=signal_names, 
                            repetitions = 10,
                            verbosity=4,
                            parallelize=True)

    start = time.time()
    my_results = my_benchmark.run_test() # Run the test. my_results is a nested dictionary with the results for each of the variables of the simulation.
    end= time.time()
    print("The time of execution of above program is :", end-start)
    df = my_benchmark.get_results_as_df() # This formats the results on a DataFrame
    print(df)
    # my_benchmark.signal_ids
    my_benchmark.save_to_file()

    results_interpreter = ResultsInterpreter(my_benchmark)
    results_interpreter.write_to_file()
    # output_string = results_interpreter.get_table_means()
    # with open('RESULTS.md', 'w') as f:
    #     f.write(output_string)


  