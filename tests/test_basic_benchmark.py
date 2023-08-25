from benchmark_tools.Benchmark import Benchmark
# import numpy as np
# import pytest

# def a_method(noisy_signal):
#     # Dummy method for testing QRF function of the benchmark.
#     results = noisy_signal # Simply return the same noisy signals.
#     return results


# # Test QRF computation of the benchmark.
# def test_benchmark_qrf():
#     # Create a dictionary of the methods to test.
#     my_methods = {
#         'Method 1': a_method, 
#     }
#     SNRin= [0, 5, 10, 20, 30, 50]
#     print(SNRin)
#     benchmark = Benchmark(task = 'denoising',
#                         methods = my_methods,
#                         N = 256, 
#                         SNRin = SNRin[::-1], 
#                         repetitions = 30,
#                         using_signals=['LinearChirp', 'CosChirp',],
#                         verbosity=0)
                        
#     benchmark.run_test()
#     results_df = benchmark.get_results_as_df() # Formats the results on a DataFrame
#     results_df = results_df.iloc[:,-1:-(len(SNRin)+1):-1].to_numpy()
#     snr_est = np.mean(results_df,axis=0)
#     snr_error = abs(np.array(SNRin)-snr_est)
#     assert np.all(snr_error<0.1), 'The noise addition is not calibrated.'

# test_benchmark_qrf()