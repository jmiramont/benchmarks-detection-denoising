# Benchmark Report

## Configuration   [Get .csv file] 

Length of signals: 512

Repetitions: 200

SNRin values: 
-5, 
0, 
5, 
10, 


### Methods  

* monte_carlo_test 

* global_mad_test 

* global_rank_env_test 

### Signals  

* LinearChirp 

## Mean results tables: 

Results shown here are the mean and standard deviation of                             the estimated detection power.                             Best performances are **bolded**. 
### Signal: LinearChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/detection/figures/html/plot_LinearChirp.html)    [[Get .csv]](/results/detection/csv_files/results_LinearChirp.csv)
|    | Method + Param                                                                                                    | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=5dB (mean)   |   SNRin=5dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) |
|---:|:------------------------------------------------------------------------------------------------------------------|:--------------------|-------------------:|:-------------------|------------------:|:-------------------|------------------:|:--------------------|-------------------:|
|  0 | monte_carlo_test{'statistic': 'Frs', 'pnorm': 2, 'rmax': 0.5, 'MC_reps': 2499}                                    | 0.06                |               0.23 | 0.09               |              0.28 | 0.17               |              0.37 | 0.34                |               0.48 |
|  1 | monte_carlo_test{'statistic': 'Frs_vs', 'pnorm': 2, 'rmax': 0.5, 'MC_reps': 2499}                                 | 0.06                |               0.24 | 0.09               |              0.29 | 0.15               |              0.36 | 0.34                |               0.47 |
|  2 | monte_carlo_test{'statistic': 'Frs', 'pnorm': 2, 'rmax': 1.0, 'MC_reps': 2499}                                    | 0.07                |               0.26 | 0.21               |              0.41 | 0.84               |              0.37 | **1.00**            |               0.00 |
|  3 | monte_carlo_test{'statistic': 'Frs_vs', 'pnorm': 2, 'rmax': 1.0, 'MC_reps': 2499}                                 | 0.07                |               0.26 | 0.41               |              0.49 | 0.99               |              0.07 | 1.00                |               0.00 |
|  4 | monte_carlo_test{'statistic': 'Frs', 'pnorm': 2, 'rmax': 2.0, 'MC_reps': 2499}                                    | 0.07                |               0.26 | 0.22               |              0.41 | 0.81               |              0.39 | 1.00                |               0.00 |
|  5 | monte_carlo_test{'statistic': 'Frs_vs', 'pnorm': 2, 'rmax': 2.0, 'MC_reps': 2499}                                 | 0.07                |               0.26 | 0.41               |              0.49 | **1.00**           |              0.00 | 1.00                |               0.00 |
|  6 | monte_carlo_test{'statistic': 'Frs', 'pnorm': 2, 'rmax': 1.0, 'MC_reps': 199}                                     | 0.08                |               0.27 | 0.21               |              0.41 | 0.82               |              0.38 | 1.00                |               0.00 |
|  7 | monte_carlo_test{'statistic': 'Frs_vs', 'pnorm': 2, 'rmax': 1.0, 'MC_reps': 199}                                  | 0.08                |               0.27 | 0.41               |              0.49 | 1.00               |              0.00 | 1.00                |               0.00 |
|  8 | global_mad_test{'statistic': 'Frs', 'MC_reps': 2499}                                                              | 0.07                |               0.26 | 0.20               |              0.40 | 0.80               |              0.40 | 1.00                |               0.00 |
|  9 | global_mad_test{'statistic': 'Frs_vs', 'MC_reps': 2499}                                                           | 0.09                |               0.28 | 0.41               |              0.49 | 1.00               |              0.00 | 1.00                |               0.00 |
| 10 | global_mad_test{'statistic': 'Frs', 'MC_reps': 199}                                                               | 0.07                |               0.26 | 0.21               |              0.41 | 0.79               |              0.41 | 1.00                |               0.00 |
| 11 | global_mad_test{'statistic': 'Frs_vs', 'MC_reps': 199}                                                            | 0.08                |               0.27 | 0.41               |              0.49 | 1.00               |              0.00 | 1.00                |               0.00 |
| 12 | global_rank_env_test{'fun': 'Fest', 'correction': 'rs'}                                                           | 0.06                |               0.23 | 0.30               |              0.46 | 0.99               |              0.07 | 1.00                |               0.00 |
| 13 | global_rank_env_test{'fun': 'Fest', 'correction': 'rs', 'rmin': 0.65, 'rmax': 1.05}                               | **0.11**            |               0.31 | **0.49**           |              0.50 | 1.00               |              0.00 | 1.00                |               0.00 |
| 14 | global_rank_env_test{'fun': 'Fest', 'correction': 'rs', 'transform': 'asin(sqrt(.))'}                             | 0.06                |               0.23 | 0.30               |              0.46 | 0.99               |              0.07 | 1.00                |               0.00 |
| 15 | global_rank_env_test{'fun': 'Fest', 'correction': 'rs', 'rmin': 0.65, 'rmax': 1.05, 'transform': 'asin(sqrt(.))'} | 0.11                |               0.31 | 0.49               |              0.50 | 1.00               |              0.00 | 1.00                |               0.00 |
