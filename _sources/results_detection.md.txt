# Benchmark Report

## Configuration

Length of signals: 256

Repetitions: 2000

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
### Signal: LinearChirp[[View Plot]](https://jmiramont.github.io/benchmarks-detection-denoising/results/detection/plot_LinearChirp.html)    [[Get .csv]](https://jmiramont.github.io/benchmarks-detection-denoising/results/detection/results_LinearChirp.csv)
|    | Method + Param                                                                                                    | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=5dB (mean)   |   SNRin=5dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) |
|---:|:------------------------------------------------------------------------------------------------------------------|:--------------------|-------------------:|:-------------------|------------------:|:-------------------|------------------:|:--------------------|-------------------:|
|  0 | monte_carlo_test{'statistic': 'Frs', 'pnorm': 2, 'rmax': 0.5, 'MC_reps': 2499}                                    | 0.05                |               0.21 | 0.07               |              0.26 | 0.14               |              0.35 | 0.29                |               0.45 |
|  1 | monte_carlo_test{'statistic': 'Frs_vs', 'pnorm': 2, 'rmax': 0.5, 'MC_reps': 2499}                                 | 0.04                |               0.21 | 0.07               |              0.26 | 0.14               |              0.34 | 0.28                |               0.45 |
|  2 | monte_carlo_test{'statistic': 'Frs', 'pnorm': 2, 'rmax': 1.0, 'MC_reps': 2499}                                    | 0.06                |               0.23 | 0.17               |              0.37 | 0.65               |              0.48 | 1.00                |               0.02 |
|  3 | monte_carlo_test{'statistic': 'Frs_vs', 'pnorm': 2, 'rmax': 1.0, 'MC_reps': 2499}                                 | 0.07                |               0.26 | 0.27               |              0.45 | 0.94               |              0.24 | **1.00**            |               0.00 |
|  4 | monte_carlo_test{'statistic': 'Frs', 'pnorm': 2, 'rmax': 2.0, 'MC_reps': 2499}                                    | 0.06                |               0.24 | 0.17               |              0.38 | 0.65               |              0.48 | 1.00                |               0.00 |
|  5 | monte_carlo_test{'statistic': 'Frs_vs', 'pnorm': 2, 'rmax': 2.0, 'MC_reps': 2499}                                 | 0.07                |               0.26 | 0.27               |              0.45 | 0.94               |              0.24 | 1.00                |               0.00 |
|  6 | global_mad_test{'statistic': 'Frs', 'MC_reps': 2499}                                                              | 0.06                |               0.23 | 0.16               |              0.37 | 0.65               |              0.48 | 1.00                |               0.00 |
|  7 | global_mad_test{'statistic': 'Frs_vs', 'MC_reps': 2499}                                                           | 0.07                |               0.25 | 0.27               |              0.44 | 0.96               |              0.19 | 1.00                |               0.00 |
|  8 | global_rank_env_test{'fun': 'Fest', 'correction': 'rs'}                                                           | 0.07                |               0.25 | 0.20               |              0.40 | 0.91               |              0.29 | 1.00                |               0.00 |
|  9 | global_rank_env_test{'fun': 'Fest', 'correction': 'rs', 'rmin': 0.65, 'rmax': 1.05}                               | **0.09**            |               0.29 | **0.34**           |              0.47 | **0.97**           |              0.16 | 1.00                |               0.00 |
| 10 | global_rank_env_test{'fun': 'Fest', 'correction': 'rs', 'transform': 'asin(sqrt(.))'}                             | 0.06                |               0.24 | 0.20               |              0.40 | 0.91               |              0.29 | 1.00                |               0.00 |
| 11 | global_rank_env_test{'fun': 'Fest', 'correction': 'rs', 'rmin': 0.65, 'rmax': 1.05, 'transform': 'asin(sqrt(.))'} | 0.09                |               0.29 | 0.34               |              0.47 | 0.97               |              0.17 | 1.00                |               0.00 |
