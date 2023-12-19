# Benchmark Report

## Configuration

Length of signals: 1024

Repetitions: 100

SNRin values: 
-5, 
0, 
10, 
20, 


### Methods  

* brevdo_method 

* contour_filtering 

* delaunay_triangulation 

* empty_space 

* thresholding_garrote 

* thresholding_hard 

* pseudo_bayesian_method 

* sz_classification 

### Signals  

* McMultiLinear 

* McMultiLinear2 

* McSyntheticMixture 

* McSyntheticMixture2 

* McSyntheticMixture3 

* HermiteFunction 

## Mean results tables: 

Results shown here are the mean and standard deviation of                               the Quality Reconstruction Factor.                               Best performances are **bolded**. 
### Signal: McMultiLinear[[View Plot]](https://jmiramont.github.io/benchmarks-detection-denoising/results/denoising_CC/plot_McMultiLinear.html)    [[Get .csv]](https://jmiramont.github.io/benchmarks-detection-denoising/results/denoising_CC/results_McMultiLinear.csv)
|    | Method + Param                                                   | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) | SNRin=20dB (mean)   |   SNRin=20dB (std) |
|---:|:-----------------------------------------------------------------|:--------------------|-------------------:|:-------------------|------------------:|:--------------------|-------------------:|:--------------------|-------------------:|
|  0 | brevdo_method                                                    | 0.49                |               0.05 | 0.76               |              0.03 | 0.97                |               0.00 | 1.00                |               0.00 |
|  1 | contour_filtering                                                | 0.30                |               0.05 | 0.60               |              0.06 | 0.98                |               0.00 | 1.00                |               0.00 |
|  2 | delaunay_triangulation                                           | 0.43                |               0.08 | 0.74               |              0.07 | 0.98                |               0.00 | 1.00                |               0.00 |
|  3 | empty_space                                                      | 0.47                |               0.08 | 0.75               |              0.06 | 0.98                |               0.00 | 1.00                |               0.00 |
|  4 | thresholding_garrote                                             | **0.58**            |               0.03 | **0.86**           |              0.01 | 0.99                |               0.00 | 1.00                |               0.00 |
|  5 | thresholding_hard                                                | 0.17                |               0.06 | 0.53               |              0.04 | **0.99**            |               0.00 | **1.00**            |               0.00 |
|  6 | pseudo_bayesian_method([], [], [], 0.4, 0.4, [], [], [], [], []) | 0.52                |               0.05 | 0.82               |              0.05 | 0.98                |               0.01 | 1.00                |               0.00 |
|  7 | pseudo_bayesian_method([], [], [], 0.4, 0.2, [], [], [], [], []) | 0.52                |               0.03 | 0.82               |              0.02 | 0.98                |               0.00 | 1.00                |               0.00 |
|  8 | sz_classification                                                | 0.51                |               0.06 | 0.83               |              0.02 | 0.98                |               0.00 | 1.00                |               0.00 |
### Signal: McMultiLinear2[[View Plot]](https://jmiramont.github.io/benchmarks-detection-denoising/results/denoising_CC/plot_McMultiLinear2.html)    [[Get .csv]](https://jmiramont.github.io/benchmarks-detection-denoising/results/denoising_CC/results_McMultiLinear2.csv)
|    | Method + Param                                                   | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) | SNRin=20dB (mean)   |   SNRin=20dB (std) |
|---:|:-----------------------------------------------------------------|:--------------------|-------------------:|:-------------------|------------------:|:--------------------|-------------------:|:--------------------|-------------------:|
|  0 | brevdo_method                                                    | 0.50                |               0.04 | 0.76               |              0.03 | 0.98                |               0.00 | 1.00                |               0.00 |
|  1 | contour_filtering                                                | 0.28                |               0.04 | 0.40               |              0.03 | 0.66                |               0.02 | 0.72                |               0.01 |
|  2 | delaunay_triangulation                                           | 0.39                |               0.07 | 0.70               |              0.09 | 0.90                |               0.02 | 0.88                |               0.00 |
|  3 | empty_space                                                      | 0.44                |               0.06 | 0.71               |              0.07 | 0.91                |               0.01 | 0.89                |               0.00 |
|  4 | thresholding_garrote                                             | **0.54**            |               0.04 | **0.83**           |              0.01 | **0.98**            |               0.00 | 1.00                |               0.00 |
|  5 | thresholding_hard                                                | 0.11                |               0.07 | 0.38               |              0.05 | 0.98                |               0.00 | **1.00**            |               0.00 |
|  6 | pseudo_bayesian_method([], [], [], 0.4, 0.4, [], [], [], [], []) | 0.49                |               0.04 | 0.77               |              0.04 | 0.97                |               0.02 | 1.00                |               0.00 |
|  7 | pseudo_bayesian_method([], [], [], 0.4, 0.2, [], [], [], [], []) | 0.48                |               0.03 | 0.75               |              0.02 | 0.98                |               0.00 | 1.00                |               0.00 |
|  8 | sz_classification                                                | 0.48                |               0.08 | 0.81               |              0.03 | 0.98                |               0.00 | 1.00                |               0.00 |
### Signal: McSyntheticMixture[[View Plot]](https://jmiramont.github.io/benchmarks-detection-denoising/results/denoising_CC/plot_McSyntheticMixture.html)    [[Get .csv]](https://jmiramont.github.io/benchmarks-detection-denoising/results/denoising_CC/results_McSyntheticMixture.csv)
|    | Method + Param                                                   | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) | SNRin=20dB (mean)   |   SNRin=20dB (std) |
|---:|:-----------------------------------------------------------------|:--------------------|-------------------:|:-------------------|------------------:|:--------------------|-------------------:|:--------------------|-------------------:|
|  0 | brevdo_method                                                    | 0.49                |               0.06 | 0.75               |              0.04 | 0.95                |               0.03 | 0.99                |               0.01 |
|  1 | contour_filtering                                                | 0.32                |               0.06 | 0.63               |              0.06 | 0.94                |               0.01 | 0.96                |               0.01 |
|  2 | delaunay_triangulation                                           | 0.39                |               0.08 | 0.67               |              0.08 | 0.94                |               0.02 | 0.95                |               0.01 |
|  3 | empty_space                                                      | 0.44                |               0.07 | 0.70               |              0.05 | 0.95                |               0.01 | 0.96                |               0.01 |
|  4 | thresholding_garrote                                             | **0.59**            |               0.03 | **0.86**           |              0.01 | 0.99                |               0.00 | 1.00                |               0.00 |
|  5 | thresholding_hard                                                | 0.22                |               0.06 | 0.60               |              0.04 | **0.99**            |               0.00 | **1.00**            |               0.00 |
|  6 | pseudo_bayesian_method([], [], [], 0.4, 0.4, [], [], [], [], []) | 0.53                |               0.06 | 0.79               |              0.05 | 0.97                |               0.02 | 0.99                |               0.00 |
|  7 | pseudo_bayesian_method([], [], [], 0.4, 0.2, [], [], [], [], []) | 0.56                |               0.04 | 0.84               |              0.02 | 0.98                |               0.00 | 1.00                |               0.00 |
|  8 | sz_classification                                                | 0.54                |               0.08 | 0.83               |              0.03 | 0.99                |               0.00 | 1.00                |               0.00 |
### Signal: McSyntheticMixture2[[View Plot]](https://jmiramont.github.io/benchmarks-detection-denoising/results/denoising_CC/plot_McSyntheticMixture2.html)    [[Get .csv]](https://jmiramont.github.io/benchmarks-detection-denoising/results/denoising_CC/results_McSyntheticMixture2.csv)
|    | Method + Param                                                   | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) | SNRin=20dB (mean)   |   SNRin=20dB (std) |
|---:|:-----------------------------------------------------------------|:--------------------|-------------------:|:-------------------|------------------:|:--------------------|-------------------:|:--------------------|-------------------:|
|  0 | brevdo_method                                                    | 0.55                |               0.04 | 0.78               |              0.02 | 0.97                |               0.00 | 1.00                |               0.00 |
|  1 | contour_filtering                                                | 0.60                |               0.08 | 0.89               |              0.02 | 0.99                |               0.00 | 1.00                |               0.00 |
|  2 | delaunay_triangulation                                           | 0.50                |               0.09 | 0.79               |              0.03 | 0.98                |               0.00 | 1.00                |               0.00 |
|  3 | empty_space                                                      | 0.53                |               0.07 | 0.80               |              0.03 | 0.98                |               0.00 | 1.00                |               0.00 |
|  4 | thresholding_garrote                                             | **0.70**            |               0.03 | **0.89**           |              0.01 | 0.99                |               0.00 | 1.00                |               0.00 |
|  5 | thresholding_hard                                                | 0.51                |               0.07 | 0.89               |              0.02 | **0.99**            |               0.00 | **1.00**            |               0.00 |
|  6 | pseudo_bayesian_method([], [], [], 0.4, 0.4, [], [], [], [], []) | 0.59                |               0.06 | 0.81               |              0.04 | 0.97                |               0.02 | 0.99                |               0.01 |
|  7 | pseudo_bayesian_method([], [], [], 0.4, 0.2, [], [], [], [], []) | 0.60                |               0.03 | 0.82               |              0.01 | 0.98                |               0.00 | 1.00                |               0.00 |
|  8 | sz_classification                                                | 0.69                |               0.06 | 0.87               |              0.03 | 0.98                |               0.00 | 1.00                |               0.00 |
### Signal: McSyntheticMixture3[[View Plot]](https://jmiramont.github.io/benchmarks-detection-denoising/results/denoising_CC/plot_McSyntheticMixture3.html)    [[Get .csv]](https://jmiramont.github.io/benchmarks-detection-denoising/results/denoising_CC/results_McSyntheticMixture3.csv)
|    | Method + Param                                                   | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) | SNRin=20dB (mean)   |   SNRin=20dB (std) |
|---:|:-----------------------------------------------------------------|:--------------------|-------------------:|:-------------------|------------------:|:--------------------|-------------------:|:--------------------|-------------------:|
|  0 | brevdo_method                                                    | 0.52                |               0.05 | 0.75               |              0.03 | 0.96                |               0.02 | 0.99                |               0.01 |
|  1 | contour_filtering                                                | 0.63                |               0.08 | **0.90**           |              0.03 | 0.99                |               0.00 | 1.00                |               0.00 |
|  2 | delaunay_triangulation                                           | 0.50                |               0.08 | 0.79               |              0.03 | 0.98                |               0.00 | 1.00                |               0.00 |
|  3 | empty_space                                                      | 0.52                |               0.08 | 0.79               |              0.04 | 0.98                |               0.00 | 1.00                |               0.00 |
|  4 | thresholding_garrote                                             | **0.70**            |               0.03 | 0.90               |              0.01 | 0.99                |               0.00 | 1.00                |               0.00 |
|  5 | thresholding_hard                                                | 0.53                |               0.07 | 0.89               |              0.02 | **0.99**            |               0.00 | **1.00**            |               0.00 |
|  6 | pseudo_bayesian_method([], [], [], 0.4, 0.4, [], [], [], [], []) | 0.59                |               0.05 | 0.81               |              0.03 | 0.97                |               0.01 | 0.99                |               0.00 |
|  7 | pseudo_bayesian_method([], [], [], 0.4, 0.2, [], [], [], [], []) | 0.58                |               0.04 | 0.81               |              0.01 | 0.98                |               0.00 | 1.00                |               0.00 |
|  8 | sz_classification                                                | 0.69                |               0.05 | 0.87               |              0.04 | 0.98                |               0.00 | 1.00                |               0.00 |
### Signal: HermiteFunction[[View Plot]](https://jmiramont.github.io/benchmarks-detection-denoising/results/denoising_CC/plot_HermiteFunction.html)    [[Get .csv]](https://jmiramont.github.io/benchmarks-detection-denoising/results/denoising_CC/results_HermiteFunction.csv)
|    | Method + Param                                                   | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) | SNRin=20dB (mean)   |   SNRin=20dB (std) |
|---:|:-----------------------------------------------------------------|:--------------------|-------------------:|:-------------------|------------------:|:--------------------|-------------------:|:--------------------|-------------------:|
|  0 | brevdo_method                                                    | 0.56                |               0.11 | 0.70               |              0.06 | 0.74                |               0.04 | 0.73                |               0.03 |
|  1 | contour_filtering                                                | 0.65                |               0.07 | 0.75               |              0.04 | 0.80                |               0.01 | 0.81                |               0.01 |
|  2 | delaunay_triangulation                                           | 0.57                |               0.10 | 0.85               |              0.05 | 0.99                |               0.00 | 1.00                |               0.00 |
|  3 | empty_space                                                      | 0.57                |               0.09 | 0.84               |              0.04 | 0.99                |               0.00 | 1.00                |               0.00 |
|  4 | thresholding_garrote                                             | 0.76                |               0.03 | 0.91               |              0.01 | 0.99                |               0.00 | 1.00                |               0.00 |
|  5 | thresholding_hard                                                | **0.88**            |               0.05 | **0.98**           |              0.00 | **1.00**            |               0.00 | **1.00**            |               0.00 |
|  6 | pseudo_bayesian_method([], [], [], 0.4, 0.4, [], [], [], [], []) | 0.33                |               0.27 | 0.46               |              0.30 | 0.65                |               0.25 | 0.77                |               0.01 |
|  7 | pseudo_bayesian_method([], [], [], 0.4, 0.2, [], [], [], [], []) | 0.15                |               0.25 | 0.14               |              0.27 | 0.04                |               0.17 | 0.00                |               0.00 |
|  8 | sz_classification                                                | 0.77                |               0.09 | 0.91               |              0.04 | 0.99                |               0.00 | 1.00                |               0.00 |
