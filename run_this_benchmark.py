import importlib
from methods import *
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


for method_instance in list_of_methods:
    method_id = method_instance.get_method_id()
    dictionary_of_methods[method_id] = method_instance.method
    dictionary_of_parameters[method_id] = method_instance.get_parameters()

print(dictionary_of_methods)
print(dictionary_of_parameters)

SNRin = [10, 20, 30]
signal_names = ['linearChirp', 'cosChirp', 'multiComponentPureTones']
my_benchmark = Benchmark(task = 'denoising',
                        methods = dictionary_of_methods,
                        N = 256,
                        parameters = dictionary_of_parameters, 
                        SNRin = SNRin,
                        using_signals=signal_names, 
                        repetitions = 5,
                        verbosity=4)
my_results = my_benchmark.run_test() # Run the test. my_results is a nested dictionary with the results for each of the variables of the simulation.
df = my_benchmark.get_results_as_df() # This formats the results on a DataFrame
print(df)
# my_benchmark.save_to_file()

results_interpreter = ResultsInterpreter(my_benchmark)
output_string = results_interpreter.write_to_file()
with open('RESULTS.md', 'a') as f:
    f.write(output_string)



# print(my_benchmark.methods_and_params_dic)


  