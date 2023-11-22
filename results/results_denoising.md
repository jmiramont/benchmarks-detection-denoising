# Benchmark Report

## Configuration   [Get .csv file] 

Length of signals: 1024

Repetitions: 30

SNRin values: 
-5, 
0, 
10, 
20, 
30, 


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

* McSyntheticMixture4 

* HermiteFunction 

## Mean results tables: 

Results shown here are the mean and standard deviation of                               the Quality Reconstruction Factor.                               Best performances are **bolded**. 
### Signal: McMultiLinear  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McMultiLinear.html)    [[Get .csv]](/results/denoising/csv_files/results_McMultiLinear.csv)
|    | Method + Param         | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) | SNRin=20dB (mean)   |   SNRin=20dB (std) | SNRin=30dB (mean)   |   SNRin=30dB (std) |
|---:|:-----------------------|:--------------------|-------------------:|:-------------------|------------------:|:--------------------|-------------------:|:--------------------|-------------------:|:--------------------|-------------------:|
|  0 | brevdo_method          | -2.34               |               0.31 | 3.25               |              0.24 | 12.96               |               0.18 | 22.45               |               0.15 | 32.01               |               0.15 |
|  1 | contour_filtering      | -0.49               |               0.25 | 1.56               |              0.33 | 13.11               |               0.18 | 22.33               |               0.19 | 31.36               |               0.58 |
|  2 | delaunay_triangulation | -1.55               |               0.82 | 2.10               |              0.66 | 14.13               |               0.58 | 24.31               |               0.27 | 33.89               |               0.29 |
|  3 | empty_space            | -2.07               |               0.78 | 1.99               |              0.75 | 13.93               |               0.55 | 24.03               |               0.26 | 33.90               |               0.35 |
|  4 | thresholding_garrote   | **0.48**            |               0.38 | **5.63**           |              0.39 | **15.73**           |               0.34 | 25.16               |               0.26 | 34.51               |               0.23 |
|  5 | thresholding_hard      | 0.11                |               0.09 | 1.21               |              0.27 | 15.64               |               0.45 | **25.65**           |               0.34 | **34.99**           |               0.26 |
|  6 | pseudo_bayesian_method | -4.11               |               0.10 | 1.09               |              0.29 | 11.69               |               0.13 | 21.66               |               0.26 | 30.09               |               3.51 |
|  7 | sz_classification      | -0.36               |               1.21 | 4.43               |              1.11 | 13.55               |               1.09 | 22.59               |               0.54 | 31.68               |               0.37 |
### Signal: McMultiLinear2  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McMultiLinear2.html)    [[Get .csv]](/results/denoising/csv_files/results_McMultiLinear2.csv)
|    | Method + Param         | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) | SNRin=20dB (mean)   |   SNRin=20dB (std) | SNRin=30dB (mean)   |   SNRin=30dB (std) |
|---:|:-----------------------|:--------------------|-------------------:|:-------------------|------------------:|:--------------------|-------------------:|:--------------------|-------------------:|:--------------------|-------------------:|
|  0 | brevdo_method          | -3.18               |               0.27 | 2.02               |              0.58 | 13.42               |               0.29 | 23.16               |               0.50 | 32.64               |               0.92 |
|  1 | contour_filtering      | -0.65               |               0.28 | 0.61               |              0.17 | 2.39                |               0.20 | 3.12                |               0.17 | 3.23                |               0.16 |
|  2 | delaunay_triangulation | -1.87               |               0.67 | 1.27               |              0.32 | 7.50                |               0.40 | 6.57                |               0.18 | 6.58                |               0.17 |
|  3 | empty_space            | -2.38               |               0.63 | 1.15               |              0.40 | 7.88                |               0.43 | 6.88                |               0.16 | 6.93                |               0.20 |
|  4 | thresholding_garrote   | **0.08**            |               0.25 | **4.71**           |              0.36 | **14.85**           |               0.30 | 24.34               |               0.29 | 33.89               |               0.28 |
|  5 | thresholding_hard      | 0.05                |               0.04 | 0.55               |              0.16 | 13.88               |               0.33 | **24.42**           |               0.26 | **34.14**           |               0.26 |
|  6 | pseudo_bayesian_method | -4.56               |               0.07 | 0.75               |              0.16 | 11.36               |               0.58 | 20.86               |               1.72 | 29.19               |               3.91 |
|  7 | sz_classification      | -0.51               |               0.72 | 3.66               |              0.31 | 13.28               |               0.77 | 20.72               |               2.37 | 21.81               |               5.03 |
### Signal: McSyntheticMixture  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McSyntheticMixture.html)    [[Get .csv]](/results/denoising/csv_files/results_McSyntheticMixture.csv)
|    | Method + Param         | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) | SNRin=20dB (mean)   |   SNRin=20dB (std) | SNRin=30dB (mean)   |   SNRin=30dB (std) |
|---:|:-----------------------|:--------------------|-------------------:|:-------------------|------------------:|:--------------------|-------------------:|:--------------------|-------------------:|:--------------------|-------------------:|
|  0 | brevdo_method          | -1.75               |               0.36 | 2.87               |              0.62 | 9.55                |               1.63 | 16.35               |               2.56 | 18.61               |               1.41 |
|  1 | contour_filtering      | -0.13               |               0.25 | 2.01               |              0.64 | 9.50                |               0.93 | 11.32               |               1.10 | 12.21               |               0.62 |
|  2 | delaunay_triangulation | -1.68               |               0.70 | 1.56               |              0.42 | 9.35                |               0.94 | 10.42               |               0.89 | 10.45               |               0.34 |
|  3 | empty_space            | -2.22               |               0.64 | 1.63               |              0.55 | 10.20               |               0.67 | 11.46               |               0.69 | 11.26               |               0.29 |
|  4 | thresholding_garrote   | **0.49**            |               0.34 | **5.48**           |              0.41 | 15.50               |               0.35 | 24.89               |               0.29 | 34.24               |               0.25 |
|  5 | thresholding_hard      | 0.17                |               0.11 | 1.63               |              0.29 | **15.51**           |               0.39 | **25.16**           |               0.32 | **34.34**           |               0.29 |
|  6 | pseudo_bayesian_method | -3.10               |               0.20 | 2.42               |              0.44 | 12.79               |               0.54 | 21.88               |               1.32 | 20.41               |               2.67 |
|  7 | sz_classification      | -0.26               |               1.01 | 4.17               |              0.70 | 15.27               |               0.39 | 24.38               |               0.35 | 29.00               |              10.48 |
### Signal: McSyntheticMixture2  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McSyntheticMixture2.html)    [[Get .csv]](/results/denoising/csv_files/results_McSyntheticMixture2.csv)
|    | Method + Param         | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) | SNRin=20dB (mean)   |   SNRin=20dB (std) | SNRin=30dB (mean)   |   SNRin=30dB (std) |
|---:|:-----------------------|:--------------------|-------------------:|:-------------------|------------------:|:--------------------|-------------------:|:--------------------|-------------------:|:--------------------|-------------------:|
|  0 | brevdo_method          | -2.15               |               0.27 | 3.01               |              0.34 | 13.08               |               0.33 | 22.86               |               1.44 | 32.82               |               2.21 |
|  1 | contour_filtering      | **1.42**            |               0.51 | **6.60**           |              0.71 | 15.59               |               0.40 | 24.57               |               0.34 | 33.51               |               0.72 |
|  2 | delaunay_triangulation | -1.22               |               0.98 | 3.39               |              0.82 | 14.95               |               0.48 | 24.52               |               0.65 | 34.30               |               0.43 |
|  3 | empty_space            | -1.83               |               1.00 | 3.34               |              1.14 | 14.62               |               0.39 | 24.70               |               0.48 | 34.50               |               0.30 |
|  4 | thresholding_garrote   | 1.39                |               0.37 | 6.49               |              0.41 | 16.36               |               0.41 | 25.99               |               0.35 | **35.52**           |               0.29 |
|  5 | thresholding_hard      | 1.10                |               0.42 | 6.33               |              0.75 | **16.91**           |               0.53 | **26.25**           |               0.36 | 35.27               |               0.29 |
|  6 | pseudo_bayesian_method | -3.56               |               0.17 | 1.75               |              0.23 | 11.47               |               1.01 | 21.71               |               1.21 | 16.74               |               1.61 |
|  7 | sz_classification      | 0.89                |               1.81 | 5.40               |              1.84 | 14.90               |               0.73 | 24.15               |               0.65 | 32.32               |               1.64 |
### Signal: McSyntheticMixture3  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McSyntheticMixture3.html)    [[Get .csv]](/results/denoising/csv_files/results_McSyntheticMixture3.csv)
|    | Method + Param         | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) | SNRin=20dB (mean)   |   SNRin=20dB (std) | SNRin=30dB (mean)   |   SNRin=30dB (std) |
|---:|:-----------------------|:--------------------|-------------------:|:-------------------|------------------:|:--------------------|-------------------:|:--------------------|-------------------:|:--------------------|-------------------:|
|  0 | brevdo_method          | -2.29               |               0.30 | 2.57               |              0.39 | 11.17               |               1.38 | 17.38               |               2.64 | 19.07               |               2.86 |
|  1 | contour_filtering      | **1.51**            |               0.65 | **7.19**           |              0.91 | 16.91               |               0.56 | 26.20               |               0.47 | 34.95               |               0.61 |
|  2 | delaunay_triangulation | -1.37               |               1.24 | 3.42               |              1.00 | 15.31               |               0.56 | 24.91               |               0.44 | 33.71               |               3.34 |
|  3 | empty_space            | -1.92               |               1.21 | 3.25               |              1.26 | 14.90               |               0.52 | 25.24               |               0.47 | 34.85               |               0.37 |
|  4 | thresholding_garrote   | 1.44                |               0.48 | 6.57               |              0.45 | 16.53               |               0.43 | 26.23               |               0.39 | 35.77               |               0.33 |
|  5 | thresholding_hard      | 1.27                |               0.39 | 6.72               |              0.58 | **17.87**           |               0.44 | **27.13**           |               0.39 | **36.33**           |               0.31 |
|  6 | pseudo_bayesian_method | -3.59               |               0.20 | 1.80               |              0.32 | 11.40               |               0.85 | 20.27               |               0.99 | 26.29               |               0.95 |
|  7 | sz_classification      | 0.94                |               1.80 | 5.73               |              2.00 | 15.27               |               1.12 | 24.78               |               0.81 | 34.41               |               0.70 |
### Signal: McSyntheticMixture4  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McSyntheticMixture4.html)    [[Get .csv]](/results/denoising/csv_files/results_McSyntheticMixture4.csv)
|    | Method + Param         | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) | SNRin=20dB (mean)   |   SNRin=20dB (std) | SNRin=30dB (mean)   |   SNRin=30dB (std) |
|---:|:-----------------------|:--------------------|-------------------:|:-------------------|------------------:|:--------------------|-------------------:|:--------------------|-------------------:|:--------------------|-------------------:|
|  0 | brevdo_method          | -4.56               |               0.09 | 0.36               |              0.07 | 10.31               |               0.17 | 20.78               |               0.49 | 31.09               |               2.29 |
|  1 | contour_filtering      | -0.75               |               0.39 | 3.47               |              0.41 | 13.14               |               0.22 | 22.77               |               0.25 | 30.27               |               1.10 |
|  2 | delaunay_triangulation | -1.52               |               0.92 | 2.35               |              0.46 | 13.72               |               0.41 | 23.58               |               0.22 | 32.04               |               0.36 |
|  3 | empty_space            | -2.01               |               0.94 | 2.18               |              0.59 | 13.68               |               0.39 | 23.51               |               0.23 | 32.64               |               0.22 |
|  4 | thresholding_garrote   | **0.57**            |               0.30 | **5.46**           |              0.31 | **15.42**           |               0.32 | **24.59**           |               0.27 | 33.68               |               0.23 |
|  5 | thresholding_hard      | 0.22                |               0.12 | 1.84               |              0.33 | 13.78               |               0.42 | 24.55               |               0.27 | **33.71**           |               0.24 |
|  6 | pseudo_bayesian_method | -4.97               |               0.06 | 0.05               |              0.08 | 9.70                |               0.91 | 17.43               |               4.10 | 19.37               |               9.13 |
|  7 | sz_classification      | -0.23               |               1.10 | 3.80               |              0.71 | 13.24               |               0.69 | 22.80               |               0.36 | 31.34               |               4.65 |
### Signal: HermiteFunction  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_HermiteFunction.html)    [[Get .csv]](/results/denoising/csv_files/results_HermiteFunction.csv)
|    | Method + Param         | SNRin=-5dB (mean)   |   SNRin=-5dB (std) | SNRin=0dB (mean)   |   SNRin=0dB (std) | SNRin=10dB (mean)   |   SNRin=10dB (std) | SNRin=20dB (mean)   |   SNRin=20dB (std) | SNRin=30dB (mean)   |   SNRin=30dB (std) |
|---:|:-----------------------|:--------------------|-------------------:|:-------------------|------------------:|:--------------------|-------------------:|:--------------------|-------------------:|:--------------------|-------------------:|
|  0 | brevdo_method          | 1.30                |               0.50 | 2.24               |              0.53 | 2.39                |               0.40 | 2.14                |               0.18 | 2.11                |               0.08 |
|  1 | contour_filtering      | 2.48                |               0.74 | 3.84               |              0.53 | 4.45                |               0.25 | 4.55                |               0.12 | 4.57                |               0.07 |
|  2 | delaunay_triangulation | -0.73               |               1.19 | 5.00               |              1.63 | 19.34               |               1.33 | 30.41               |               1.04 | 40.57               |               0.85 |
|  3 | empty_space            | -1.35               |               1.37 | 4.68               |              1.88 | 18.10               |               1.45 | 29.73               |               1.12 | 40.39               |               0.98 |
|  4 | thresholding_garrote   | 1.99                |               0.46 | 7.05               |              0.46 | 17.11               |               0.45 | 27.13               |               0.45 | 37.12               |               0.44 |
|  5 | thresholding_hard      | **6.17**            |               1.69 | **14.19**          |              0.87 | **23.47**           |               0.81 | **32.54**           |               0.78 | **41.83**           |               0.74 |
|  6 | pseudo_bayesian_method | -1.21               |               0.88 | 0.89               |              1.75 | 2.93                |               2.53 | 5.17                |               1.75 | 6.78                |               1.18 |
|  7 | sz_classification      | 2.38                |               3.21 | 6.68               |              2.35 | 16.74               |               1.60 | 27.49               |               1.71 | 37.03               |               1.23 |
