# Benchmark Report

## Configuration   [Get .csv file] 

Length of signals: 1024

Repetitions: 100

SNRin values: 
-5, 
0, 
10, 
20, 


### Methods  

* contour_filtering 

* delaunay_triangulation 

* empty_space 

* thresholding_garrote 

* thresholding_hard 

* sz_classification 

* brevdo_method 

* pseudo_bayesian_method 

### Signals  

* McMultiLinear 

* McMultiLinear2 

* McSyntheticMixture 

* McSyntheticMixture2 

* McSyntheticMixture3 

* McSyntheticMixture4 

* HermiteFunction 

## Mean results tables: 

Results shown here are the mean and standard deviation of                               the Quality Reconstruction Factor.                               Best performances are **bolded**. 
### Signal: McMultiLinear  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McMultiLinear.html)    [[Get .csv]](/results/denoising/csv_files/results_McMultiLinear.csv)
|    | Method + Param                                                   | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) | SNRin=20dB (mean)   |   SNRin=20dB (std) |
|---:|:-----------------------------------------------------------------|:--------------------|-------------------:|:-------------------|------------------:|:--------------------|-------------------:|:--------------------|-------------------:|
|  0 | brevdo_method                                                    | -3.46               |               0.23 | 1.57               |              0.28 | 11.65               |               0.12 | 21.62               |               0.12 |
|  1 | contour_filtering                                                | -0.51               |               0.22 | 1.60               |              0.40 | 13.09               |               0.19 | 22.32               |               0.18 |
|  2 | delaunay_triangulation                                           | -1.52               |               0.78 | 2.13               |              0.54 | 14.19               |               0.51 | 24.30               |               0.29 |
|  3 | empty_space                                                      | -2.01               |               0.77 | 1.97               |              0.67 | 13.94               |               0.56 | 24.05               |               0.30 |
|  4 | thresholding_garrote                                             | **0.45**            |               0.34 | **5.62**           |              0.39 | **15.70**           |               0.34 | 25.12               |               0.29 |
|  5 | thresholding_hard                                                | 0.11                |               0.08 | 1.18               |              0.27 | 15.67               |               0.39 | **25.64**           |               0.33 |
|  6 | pseudo_bayesian_method([], [], [], 0.4, 0.4, [], [], [], [], []) | -2.47               |               0.33 | 3.49               |              0.71 | 13.90               |               1.14 | 24.04               |               1.11 |
|  7 | pseudo_bayesian_method([], [], [], 0.4, 0.2, [], [], [], [], []) | -2.86               |               0.22 | 3.27               |              0.40 | 14.29               |               0.26 | 24.41               |               0.24 |
|  8 | sz_classification                                                | -0.46               |               1.14 | 4.33               |              1.04 | 13.37               |               0.97 | 22.49               |               0.33 |
### Signal: McMultiLinear2  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McMultiLinear2.html)    [[Get .csv]](/results/denoising/csv_files/results_McMultiLinear2.csv)
|    | Method + Param                                                   | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) | SNRin=20dB (mean)   |   SNRin=20dB (std) |
|---:|:-----------------------------------------------------------------|:--------------------|-------------------:|:-------------------|------------------:|:--------------------|-------------------:|:--------------------|-------------------:|
|  0 | brevdo_method                                                    | -3.77               |               0.28 | 1.60               |              0.45 | 12.96               |               0.24 | 22.99               |               0.20 |
|  1 | contour_filtering                                                | -0.62               |               0.27 | 0.62               |              0.16 | 2.47                |               0.21 | 3.18                |               0.17 |
|  2 | delaunay_triangulation                                           | -1.85               |               0.75 | 1.29               |              0.32 | 7.38                |               0.62 | 6.59                |               0.12 |
|  3 | empty_space                                                      | -2.38               |               0.68 | 1.14               |              0.36 | 7.81                |               0.62 | 6.90                |               0.12 |
|  4 | thresholding_garrote                                             | **0.14**            |               0.30 | **4.79**           |              0.29 | **14.89**           |               0.26 | 24.37               |               0.25 |
|  5 | thresholding_hard                                                | 0.05                |               0.05 | 0.51               |              0.16 | 13.97               |               0.34 | **24.45**           |               0.29 |
|  6 | pseudo_bayesian_method([], [], [], 0.4, 0.4, [], [], [], [], []) | -3.29               |               0.21 | 2.32               |              0.54 | 12.75               |               1.82 | 23.50               |               0.91 |
|  7 | pseudo_bayesian_method([], [], [], 0.4, 0.2, [], [], [], [], []) | -3.68               |               0.14 | 1.71               |              0.28 | 13.46               |               0.25 | 23.71               |               0.21 |
|  8 | sz_classification                                                | -0.58               |               0.85 | 3.64               |              0.40 | 13.85               |               0.48 | 23.57               |               1.01 |
### Signal: McSyntheticMixture  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McSyntheticMixture.html)    [[Get .csv]](/results/denoising/csv_files/results_McSyntheticMixture.csv)
|    | Method + Param                                                   | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) | SNRin=20dB (mean)   |   SNRin=20dB (std) |
|---:|:-----------------------------------------------------------------|:--------------------|-------------------:|:-------------------|------------------:|:--------------------|-------------------:|:--------------------|-------------------:|
|  0 | brevdo_method                                                    | -2.64               |               0.33 | 2.15               |              0.51 | 10.48               |               2.05 | 20.42               |               3.82 |
|  1 | contour_filtering                                                | -0.14               |               0.27 | 2.10               |              0.59 | 9.22                |               0.92 | 11.27               |               1.24 |
|  2 | delaunay_triangulation                                           | -1.64               |               0.74 | 1.57               |              0.40 | 9.33                |               1.01 | 10.30               |               0.89 |
|  3 | empty_space                                                      | -2.12               |               0.74 | 1.63               |              0.51 | 10.10               |               0.73 | 11.37               |               0.66 |
|  4 | thresholding_garrote                                             | **0.49**            |               0.34 | **5.49**           |              0.38 | **15.51**           |               0.36 | 24.89               |               0.31 |
|  5 | thresholding_hard                                                | 0.16                |               0.11 | 1.59               |              0.33 | 15.48               |               0.35 | **25.14**           |               0.32 |
|  6 | pseudo_bayesian_method([], [], [], 0.4, 0.4, [], [], [], [], []) | -1.62               |               0.37 | 3.36               |              0.79 | 12.66               |               1.93 | 19.19               |               1.96 |
|  7 | pseudo_bayesian_method([], [], [], 0.4, 0.2, [], [], [], [], []) | -1.87               |               0.28 | 4.09               |              0.40 | 14.46               |               0.37 | 21.09               |               0.42 |
|  8 | sz_classification                                                | -0.27               |               0.99 | 4.26               |              0.65 | 15.22               |               0.39 | 24.26               |               0.48 |
### Signal: McSyntheticMixture2  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McSyntheticMixture2.html)    [[Get .csv]](/results/denoising/csv_files/results_McSyntheticMixture2.csv)
|    | Method + Param                                                   | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) | SNRin=20dB (mean)   |   SNRin=20dB (std) |
|---:|:-----------------------------------------------------------------|:--------------------|-------------------:|:-------------------|------------------:|:--------------------|-------------------:|:--------------------|-------------------:|
|  0 | brevdo_method                                                    | -3.15               |               0.23 | 1.95               |              0.25 | 12.15               |               0.32 | 22.78               |               0.59 |
|  1 | contour_filtering                                                | 1.31                |               0.61 | 6.37               |              0.66 | 15.56               |               0.34 | 24.55               |               0.30 |
|  2 | delaunay_triangulation                                           | -1.11               |               0.85 | 3.29               |              0.77 | 14.83               |               0.54 | 24.60               |               0.56 |
|  3 | empty_space                                                      | -1.67               |               0.95 | 3.17               |              1.10 | 14.64               |               0.43 | 24.83               |               0.49 |
|  4 | thresholding_garrote                                             | **1.38**            |               0.39 | **6.47**           |              0.40 | 16.33               |               0.35 | 25.94               |               0.30 |
|  5 | thresholding_hard                                                | 1.12                |               0.41 | 6.29               |              0.64 | **16.96**           |               0.51 | **26.15**           |               0.34 |
|  6 | pseudo_bayesian_method([], [], [], 0.4, 0.4, [], [], [], [], []) | -1.83               |               0.37 | 3.19               |              0.60 | 12.51               |               1.93 | 20.02               |               3.30 |
|  7 | pseudo_bayesian_method([], [], [], 0.4, 0.2, [], [], [], [], []) | -2.30               |               0.19 | 3.14               |              0.22 | 13.58               |               0.27 | 23.10               |               0.53 |
|  8 | sz_classification                                                | 0.91                |               1.91 | 5.32               |              1.55 | 14.63               |               0.67 | 23.96               |               0.68 |
### Signal: McSyntheticMixture3  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McSyntheticMixture3.html)    [[Get .csv]](/results/denoising/csv_files/results_McSyntheticMixture3.csv)
|    | Method + Param                                                   | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) | SNRin=20dB (mean)   |   SNRin=20dB (std) |
|---:|:-----------------------------------------------------------------|:--------------------|-------------------:|:-------------------|------------------:|:--------------------|-------------------:|:--------------------|-------------------:|
|  0 | brevdo_method                                                    | -3.22               |               0.26 | 1.77               |              0.37 | 11.37               |               1.28 | 19.67               |               3.19 |
|  1 | contour_filtering                                                | **1.60**            |               0.72 | **6.93**           |              0.97 | 16.99               |               0.48 | 26.15               |               0.51 |
|  2 | delaunay_triangulation                                           | -1.22               |               0.93 | 3.50               |              0.93 | 15.26               |               0.51 | 24.93               |               0.48 |
|  3 | empty_space                                                      | -1.80               |               0.89 | 3.34               |              1.23 | 14.89               |               0.51 | 25.17               |               0.53 |
|  4 | thresholding_garrote                                             | 1.43                |               0.46 | 6.56               |              0.43 | 16.49               |               0.42 | 26.19               |               0.38 |
|  5 | thresholding_hard                                                | 1.24                |               0.44 | 6.59               |              0.72 | **17.80**           |               0.46 | **27.00**           |               0.40 |
|  6 | pseudo_bayesian_method([], [], [], 0.4, 0.4, [], [], [], [], []) | -1.73               |               0.30 | 3.42               |              0.47 | 12.73               |               0.76 | 18.60               |               2.14 |
|  7 | pseudo_bayesian_method([], [], [], 0.4, 0.2, [], [], [], [], []) | -2.37               |               0.21 | 3.05               |              0.22 | 13.39               |               0.29 | 20.81               |               0.70 |
|  8 | sz_classification                                                | 0.92                |               1.83 | 5.34               |              1.84 | 15.06               |               1.04 | 24.60               |               0.57 |
### Signal: McSyntheticMixture4  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McSyntheticMixture4.html)    [[Get .csv]](/results/denoising/csv_files/results_McSyntheticMixture4.csv)
|    | Method + Param                                                   | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) | SNRin=20dB (mean)   |   SNRin=20dB (std) |
|---:|:-----------------------------------------------------------------|:--------------------|-------------------:|:-------------------|------------------:|:--------------------|-------------------:|:--------------------|-------------------:|
|  0 | brevdo_method                                                    | -4.81               |               0.09 | 0.15               |              0.10 | 10.22               |               0.10 | 20.84               |               0.29 |
|  1 | contour_filtering                                                | -0.70               |               0.38 | 3.48               |              0.39 | 13.21               |               0.24 | 22.79               |               0.26 |
|  2 | delaunay_triangulation                                           | -1.36               |               0.80 | 2.40               |              0.47 | 13.59               |               0.48 | 23.57               |               0.26 |
|  3 | empty_space                                                      | -1.87               |               0.84 | 2.23               |              0.65 | 13.66               |               0.45 | 23.52               |               0.26 |
|  4 | thresholding_garrote                                             | **0.58**            |               0.31 | **5.45**           |              0.33 | **15.44**           |               0.31 | **24.62**           |               0.26 |
|  5 | thresholding_hard                                                | 0.21                |               0.14 | 1.81               |              0.35 | 13.77               |               0.43 | 24.52               |               0.28 |
|  6 | pseudo_bayesian_method([], [], [], 0.4, 0.4, [], [], [], [], []) | -4.50               |               0.11 | 0.85               |              0.19 | 10.45               |               0.88 | 17.99               |               3.41 |
|  7 | pseudo_bayesian_method([], [], [], 0.4, 0.2, [], [], [], [], []) | -4.81               |               0.07 | 0.33               |              0.07 | 10.66               |               0.09 | 21.05               |               0.16 |
|  8 | sz_classification                                                | -0.30               |               1.22 | 3.82               |              0.82 | 13.25               |               0.57 | 22.82               |               0.37 |
### Signal: HermiteFunction  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_HermiteFunction.html)    [[Get .csv]](/results/denoising/csv_files/results_HermiteFunction.csv)
|    | Method + Param                                                   | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) | SNRin=20dB (mean)   |   SNRin=20dB (std) |
|---:|:-----------------------------------------------------------------|:--------------------|-------------------:|:-------------------|------------------:|:--------------------|-------------------:|:--------------------|-------------------:|
|  0 | brevdo_method                                                    | 1.03                |               0.58 | 2.86               |              0.64 | 3.51                |               0.54 | 3.31                |               0.40 |
|  1 | contour_filtering                                                | 2.42                |               0.68 | 3.63               |              0.61 | 4.49                |               0.22 | 4.55                |               0.11 |
|  2 | delaunay_triangulation                                           | -0.59               |               1.11 | 5.28               |              1.52 | 19.20               |               1.25 | 30.40               |               1.01 |
|  3 | empty_space                                                      | -1.28               |               1.19 | 4.99               |              1.71 | 18.07               |               1.40 | 29.83               |               1.26 |
|  4 | thresholding_garrote                                             | 2.03                |               0.53 | 7.09               |              0.53 | 17.15               |               0.52 | 27.17               |               0.52 |
|  5 | thresholding_hard                                                | **5.96**            |               1.49 | **14.14**          |              0.93 | **23.41**           |               0.79 | **32.47**           |               0.73 |
|  6 | pseudo_bayesian_method([], [], [], 0.4, 0.4, [], [], [], [], []) | 0.11                |               1.12 | 1.64               |              1.42 | 3.13                |               1.37 | 3.94                |               0.20 |
|  7 | pseudo_bayesian_method([], [], [], 0.4, 0.2, [], [], [], [], []) | -0.59               |               0.91 | 0.22               |              1.19 | 0.16                |               0.87 | -0.00               |               0.00 |
|  8 | sz_classification                                                | 2.08                |               2.76 | 6.74               |              2.36 | 17.12               |               1.91 | 27.35               |               1.71 |
