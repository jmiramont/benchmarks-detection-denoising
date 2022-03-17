import importlib
from methods import *
from methods.MethodTemplate import MethodTemplate as MethodTemplate
import time
import inspect
# Collects all the methods in the folder/ module "methods" and make a global list of them.
modules = dir()
modules = [mod_name for mod_name in modules if mod_name.startswith('method_')]
global list_of_methods # Use with caution.

list_of_methods = list()    
for mod_name in modules:
    mod = importlib.import_module('methods.' + mod_name)
    classes_in_mod = inspect.getmembers(mod, inspect.isclass)
    for a_class in classes_in_mod:
        method_class = getattr(mod, a_class[0])
        class_parent = method_class.__bases__[0]
        if class_parent == MethodTemplate:
            list_of_methods.append(method_class())


from benchmark_demo.Benchmark import *
import numpy as np
from benchmark_demo.ResultsInterpreter import ResultsInterpreter


dictionary_of_methods = dict()
dictionary_of_parameters = dict()

# Select only those methods for detection.
for method_instance in list_of_methods:
    if method_instance.task == 'detection':
        method_id = method_instance.id
        dictionary_of_methods[method_id] = method_instance.method
        dictionary_of_parameters[method_id] = method_instance.get_parameters()

SNRin = [10, 20]

# Standard test:
signal_names = ['LinearChirp'] # Single-component signals
                # 'McCrossingChirps',                       # Crossing-components  
                # 'McHarmonic','McPureTones',]               # Multi-Component Harmonic signals  
                # 'McModulatedTones','McDoubleCosChirp',    # Multi-Component Non-Harmonic  
                # 'McSyntheticMixture','McSyntheticMixture2',
                # 'HermiteFunction','HermiteElipse',        # Hermite type signals  
                # 'ToneDumped','ToneSharpAttack',           # Dumped and Sharps attacks  
                # 'McOnOffTones']                           # Modes that born and die

if __name__ == "__main__":
    np.random.seed(0) 
    my_benchmark = Benchmark(task = 'detection',
                            methods = dictionary_of_methods,
                            N = 256,
                            parameters = dictionary_of_parameters, 
                            SNRin = SNRin,
                            using_signals=signal_names, 
                            repetitions=200,
                            verbosity=4,
                            parallelize=12)

    start = time.time()
    my_results = my_benchmark.run_test() # Run the test. my_results is a nested dictionary with the results for each of the variables of the simulation.
    end= time.time()
    print("The time of execution of above program is :", end-start)
    df = my_benchmark.get_results_as_df() # This formats the results on a DataFrame
    print(df)
    
    # Save the benchmark to a file. Notice that only the methods_ids are saved.
    my_benchmark.save_to_file(filename = 'results/last_benchmark_detection')
    # results_interpreter = ResultsInterpreter(my_benchmark)
    # results_interpreter.save_report()
    # output_string = results_interpreter.get_table_means()
    # with open('RESULTS.md', 'w') as f:
    #     f.write(output_string)


  