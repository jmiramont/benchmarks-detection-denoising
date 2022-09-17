# Benchmark Report

## Configuration   [Get .csv file] 

Length of signals: 512

Repetitions: 5

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

## Mean results tables: 

Results shown here are the mean and standard deviation of                               the Quality Reconstruction Factor.                               Best performances are **bolded**. 
### Signal: LinearChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_LinearChirp.html)    [[Get .csv]](/results/denoising/csv_files/results_LinearChirp.csv)
|    | Method + Param         | SNRin=0dB (mean)      |   SNRin=0dB (std) | SNRin=10dB (mean)      |   SNRin=10dB (std) | SNRin=20dB (mean)     |   SNRin=20dB (std) | SNRin=30dB (mean)     |   SNRin=30dB (std) |
|---:|:-----------------------|:----------------------|------------------:|:-----------------------|-------------------:|:----------------------|-------------------:|:----------------------|-------------------:|
|  0 | contour_filtering      | **2.232791615299152** |         0.296668  | 12.133546280442854     |          0.297878  | 22.686074364476106    |          0.139233  | 31.76252211361169     |           0.13038  |
|  1 | delaunay_triangulation | 2.166002590945351     |         1.21202   | **16.174203824026012** |          1.13884   | 20.288193797403675    |          3.92537   | 10.95209874887774     |           0.419816 |
|  2 | empty_space            | 2.0331656203474355    |         1.52385   | 15.584358128112664     |          0.81445   | **24.94571545565061** |          0.621927  | **34.28452804550783** |           0.384682 |
|  3 | hard_thresholding      | 0.2996407020727206    |         0.0407467 | 10.298706891177252     |          0.0530224 | 20.36189366918533     |          0.0845979 | 30.41956196360236     |           0.09992  |
### Signal: CosChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_CosChirp.html)    [[Get .csv]](/results/denoising/csv_files/results_CosChirp.csv)
|    | Method + Param         | SNRin=0dB (mean)       |   SNRin=0dB (std) | SNRin=10dB (mean)      |   SNRin=10dB (std) | SNRin=20dB (mean)      |   SNRin=20dB (std) | SNRin=30dB (mean)     |   SNRin=30dB (std) |
|---:|:-----------------------|:-----------------------|------------------:|:-----------------------|-------------------:|:-----------------------|-------------------:|:----------------------|-------------------:|
|  0 | contour_filtering      | **2.2877589510583762** |         0.425414  | 12.071946225811336     |          0.0610521 | 22.67445671237504      |          0.249706  | 32.30138579531602     |          0.235398  |
|  1 | delaunay_triangulation | 1.9702935001203088     |         0.692971  | 11.074405051033523     |          2.2573    | 21.298320601649323     |          5.54035   | 10.137838843119708    |          0.83477   |
|  2 | empty_space            | 1.857256116801301      |         0.498538  | **13.754950671078424** |          1.03448   | **24.133835793790176** |          0.323624  | **32.60053554423963** |          0.316084  |
|  3 | hard_thresholding      | 0.22236517414921475    |         0.0437812 | 10.334966269466992     |          0.0827366 | 20.36666783227637      |          0.0876093 | 30.45040781728136     |          0.0871319 |
