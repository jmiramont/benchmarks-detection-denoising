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
