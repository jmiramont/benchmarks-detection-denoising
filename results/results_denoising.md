<<<<<<< HEAD
# Benchmark Report

## Configuration   [Get .csv file] 

Length of signals: 1024

Repetitions: 30

SNRin values: 
0, 
10, 
20, 
30, 


### Methods  

* contour_filtering 

* delaunay_triangulation 

* empty_space 

* hard_thresholding 

### Signals  

* LinearChirp 

* CosChirp 

* McMultiLinear 

* McCosPlusTone 

* McMultiCos2 

* McTripleImpulse 

## Mean results tables: 

Results shown here are the mean and standard deviation of                               the Quality Reconstruction Factor.                               Best performances are **bolded**. 
### Signal: LinearChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_LinearChirp.html)    [[Get .csv]](/results/denoising/csv_files/results_LinearChirp.csv)
|    | Method + Param         | SNRin=0dB (mean)      |   SNRin=0dB (std) | SNRin=10dB (mean)      |   SNRin=10dB (std) | SNRin=20dB (mean)      |   SNRin=20dB (std) | SNRin=30dB (mean)     |   SNRin=30dB (std) |
|---:|:-----------------------|:----------------------|------------------:|:-----------------------|-------------------:|:-----------------------|-------------------:|:----------------------|-------------------:|
|  0 | contour_filtering      | 1.7174146445805976    |          0.153674 | 11.967415152989359     |           0.149003 | 22.948473258200156     |           0.172932 | 31.921121106022348    |           0.334921 |
|  1 | delaunay_triangulation | 3.365122967431782     |          1.08581  | 17.10837785435346      |           0.765657 | 26.846571201003048     |           0.716134 | 36.17706191321853     |           0.643896 |
|  2 | empty_space            | 3.988937520150123     |          0.954803 | 16.806945432163314     |           0.715555 | 26.750945612388154     |           0.641329 | 36.180967772901425    |           0.791646 |
|  3 | hard_thresholding      | **9.975734559109377** |          1.03476  | **19.748612498783977** |           0.836217 | **28.627205603129017** |           0.572435 | **37.82973690966307** |           0.564529 |
### Signal: CosChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_CosChirp.html)    [[Get .csv]](/results/denoising/csv_files/results_CosChirp.csv)
|    | Method + Param         | SNRin=0dB (mean)      |   SNRin=0dB (std) | SNRin=10dB (mean)     |   SNRin=10dB (std) | SNRin=20dB (mean)      |   SNRin=20dB (std) | SNRin=30dB (mean)     |   SNRin=30dB (std) |
|---:|:-----------------------|:----------------------|------------------:|:----------------------|-------------------:|:-----------------------|-------------------:|:----------------------|-------------------:|
|  0 | contour_filtering      | 1.7197188346406174    |          0.15927  | 11.92114224449886     |           0.123086 | 22.807677225607883     |           0.138022 | 31.801211214157938    |           0.206807 |
|  1 | delaunay_triangulation | 2.2400693442315402    |          0.741935 | 15.526012325029546    |           1.25357  | 25.999169926732698     |           0.419627 | 34.66571151270989     |           0.566782 |
|  2 | empty_space            | 2.792511305560302     |          0.922539 | 16.036957436393752    |           0.651413 | 25.881173490430903     |           0.508314 | 34.50961398190782     |           0.569347 |
|  3 | hard_thresholding      | **8.652104823325832** |          1.15042  | **18.82788715203738** |           0.669297 | **27.940316921883607** |           0.587991 | **36.93449111637964** |           0.346967 |
### Signal: McMultiLinear  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McMultiLinear.html)    [[Get .csv]](/results/denoising/csv_files/results_McMultiLinear.csv)
|    | Method + Param         | SNRin=0dB (mean)      |   SNRin=0dB (std) | SNRin=10dB (mean)    |   SNRin=10dB (std) | SNRin=20dB (mean)      |   SNRin=20dB (std) | SNRin=30dB (mean)     |   SNRin=30dB (std) |
|---:|:-----------------------|:----------------------|------------------:|:---------------------|-------------------:|:-----------------------|-------------------:|:----------------------|-------------------:|
|  0 | contour_filtering      | 1.791703323573691     |          0.160875 | 11.796940132477511   |           0.130007 | 21.805198003258845     |           0.173869 | 29.023781485488662    |           0.264431 |
|  1 | delaunay_triangulation | 1.4039964752006862    |          0.451006 | 13.37104816357005    |           0.830327 | 24.44027924985802      |           0.423403 | 33.642578575411086    |           0.280689 |
|  2 | empty_space            | 1.7496442826294716    |          0.547746 | 14.462256173167196   |           0.661038 | **24.456617063354233** |           0.432245 | **33.67150087323708** |           0.339314 |
|  3 | hard_thresholding      | **2.509136658219219** |          0.456725 | **14.6208048234701** |           0.343896 | 24.018712009265016     |           0.341121 | 33.07884939945442     |           0.30336  |
### Signal: McCosPlusTone  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McCosPlusTone.html)    [[Get .csv]](/results/denoising/csv_files/results_McCosPlusTone.csv)
|    | Method + Param         | SNRin=0dB (mean)      |   SNRin=0dB (std) | SNRin=10dB (mean)      |   SNRin=10dB (std) | SNRin=20dB (mean)      |   SNRin=20dB (std) | SNRin=30dB (mean)      |   SNRin=30dB (std) |
|---:|:-----------------------|:----------------------|------------------:|:-----------------------|-------------------:|:-----------------------|-------------------:|:-----------------------|-------------------:|
|  0 | contour_filtering      | 1.7843882441703045    |          0.185433 | 11.838152988942342     |           0.142983 | 22.181071603732605     |           0.188408 | 29.82825535888418      |           0.271726 |
|  1 | delaunay_triangulation | 1.5216977539559597    |          0.437896 | 12.25856563118482      |           1.52661  | 24.566548697578583     |           0.438048 | 33.39279493122197      |           3.38488  |
|  2 | empty_space            | 1.715330528481366     |          0.655872 | 13.99468316297572      |           1.19676  | 24.628413798804083     |           0.445957 | 33.984798801941274     |           0.355532 |
|  3 | hard_thresholding      | **3.158446630871079** |          0.715127 | **15.775774929664001** |           0.286063 | **24.894001295769534** |           0.334191 | **34.080140927984026** |           0.290665 |
### Signal: McMultiCos2  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McMultiCos2.html)    [[Get .csv]](/results/denoising/csv_files/results_McMultiCos2.csv)
|    | Method + Param         | SNRin=0dB (mean)       |   SNRin=0dB (std) | SNRin=10dB (mean)      |   SNRin=10dB (std) | SNRin=20dB (mean)      |   SNRin=20dB (std) | SNRin=30dB (mean)    |   SNRin=30dB (std) |
|---:|:-----------------------|:-----------------------|------------------:|:-----------------------|-------------------:|:-----------------------|-------------------:|:---------------------|-------------------:|
|  0 | contour_filtering      | 1.631336374001413      |          0.153992 | 11.708652923154135     |           0.14121  | 21.831203070552352     |          0.207292  | 29.079829935835917   |          0.323273  |
|  1 | delaunay_triangulation | 1.177670963887392      |          0.358331 | 6.947116457514144      |           0.736987 | 8.814354080331693      |          0.507242  | 9.232269953207812    |          0.325748  |
|  2 | empty_space            | 1.2198610634847507     |          0.379483 | 8.66723109057043       |           0.630872 | 11.03844110857969      |          0.0997661 | 11.355939910862155   |          0.0435552 |
|  3 | hard_thresholding      | **2.4812547799961817** |          0.496973 | **14.481829113836957** |           0.413907 | **23.721597421285473** |          0.229198  | **32.7319289705583** |          0.173915  |
### Signal: McTripleImpulse  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McTripleImpulse.html)    [[Get .csv]](/results/denoising/csv_files/results_McTripleImpulse.csv)
|    | Method + Param         | SNRin=0dB (mean)      |   SNRin=0dB (std) | SNRin=10dB (mean)     |   SNRin=10dB (std) | SNRin=20dB (mean)     |   SNRin=20dB (std) | SNRin=30dB (mean)   |   SNRin=30dB (std) |
|---:|:-----------------------|:----------------------|------------------:|:----------------------|-------------------:|:----------------------|-------------------:|:--------------------|-------------------:|
|  0 | contour_filtering      | 1.4209945893562597    |          0.192541 | 11.411136367828636    |           0.507638 | 18.80871219067679     |           1.25885  | 22.302692070775038  |           1.98429  |
|  1 | delaunay_triangulation | 1.593915208369202     |          0.5217   | 11.516755146064957    |           1.05993  | 22.551778010812765    |           0.717868 | 32.531472030498925  |           0.329003 |
|  2 | empty_space            | 1.7238441822538422    |          0.446879 | 12.686997083495005    |           0.736511 | 23.065885904845832    |           0.539285 | **nan**             |         nan        |
|  3 | hard_thresholding      | **6.299402730704341** |          0.684911 | **16.37177287114876** |           0.473144 | **25.57402750478404** |           0.353266 | 34.74425308555994   |           0.34222  |
=======
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
>>>>>>> dev
