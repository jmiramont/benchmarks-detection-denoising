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

* garrote_thresholding 

* hard_thresholding 

* block_thresholding 

* soft_thresholding 

### Signals  

* LinearChirp 

* CosChirp 

## Mean results tables: 

Results shown here are the mean and standard deviation of                               the Quality Reconstruction Factor.                               Best performances are **bolded**. 
### Signal: LinearChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_LinearChirp.html)    [[Get .csv]](/results/denoising/csv_files/results_LinearChirp.csv)
|    | Method + Param         | SNRin=0dB (mean)      |   SNRin=0dB (std) | SNRin=10dB (mean)      |   SNRin=10dB (std) | SNRin=20dB (mean)    |   SNRin=20dB (std) | SNRin=30dB (mean)      |   SNRin=30dB (std) |
|---:|:-----------------------|:----------------------|------------------:|:-----------------------|-------------------:|:---------------------|-------------------:|:-----------------------|-------------------:|
|  0 | contour_filtering      | 2.189395794362055     |          0.229285 | 12.349088919305585     |           0.234787 | 23.168562091084816   |           0.240161 | 31.962014141870725     |          0.345891  |
|  1 | delaunay_triangulation | 3.3534457386965877    |          0.307349 | 17.254961936228835     |           0.742889 | 26.604583973357734   |           0.577689 | 34.17032470152022      |          2.063     |
|  2 | garrote_thresholding   | 6.43445581183812      |          0.5134   | 16.389128883116836     |           0.52026  | 26.144855180468085   |           0.420707 | 35.8879980184976       |          0.508254  |
|  3 | hard_thresholding      | 5.843341187417893     |          1.33827  | **18.829851082072587** |           0.726701 | **27.9501008297765** |           0.609788 | **37.141714121033615** |          0.580027  |
|  4 | block_thresholding     | 4.873640833581871     |          0.533877 | 8.585735270484802      |           0.571147 | 9.060631154026932    |           0.213805 | 9.113689820906481      |          0.0685136 |
|  5 | soft_thresholding      | **6.490249711621369** |          0.72928  | 15.09559910028732      |           0.703561 | 24.25174659365826    |           0.592973 | 33.60435741663464      |          0.538124  |
### Signal: CosChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_CosChirp.html)    [[Get .csv]](/results/denoising/csv_files/results_CosChirp.csv)
|    | Method + Param         | SNRin=0dB (mean)      |   SNRin=0dB (std) | SNRin=10dB (mean)      |   SNRin=10dB (std) | SNRin=20dB (mean)      |   SNRin=20dB (std) | SNRin=30dB (mean)     |   SNRin=30dB (std) |
|---:|:-----------------------|:----------------------|------------------:|:-----------------------|-------------------:|:-----------------------|-------------------:|:----------------------|-------------------:|
|  0 | contour_filtering      | 2.2398933983869047    |         0.15627   | 12.355551446438982     |           0.159057 | 23.106517968308154     |           0.240108 | 31.886972413571982    |           0.327595 |
|  1 | delaunay_triangulation | 2.9009984088804615    |         0.66987   | 15.508367972138714     |           2.2579   | 26.33816068369132      |           0.45541  | 31.057317042667       |           4.59426  |
|  2 | garrote_thresholding   | **6.613130230396425** |         0.400742  | 16.426704487284926     |           0.461279 | 26.08955873325412      |           0.473025 | 35.83828061989654     |           0.499195 |
|  3 | hard_thresholding      | 5.457226810643222     |         0.93313   | **18.642071091985027** |           0.72017  | **27.665810437860433** |           0.532219 | **36.83968756819775** |           0.492348 |
|  4 | block_thresholding     | 0.013228296176536276  |         0.0185365 | 0.0                    |           0        | 0.0                    |           0        | 0.0                   |           0        |
|  5 | soft_thresholding      | 6.563265280182614     |         0.491673  | 14.969295980600837     |           0.510898 | 23.97444581555257      |           0.534931 | 33.288570972879405    |           0.508372 |
