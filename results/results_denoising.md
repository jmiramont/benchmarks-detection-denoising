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

* block_thresholding 

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
|    | Method + Param         | SNRin=0dB (mean)      |   SNRin=0dB (std) | SNRin=10dB (mean)      |   SNRin=10dB (std) | SNRin=20dB (mean)      |   SNRin=20dB (std) | SNRin=30dB (mean)     |   SNRin=30dB (std) |
|---:|:-----------------------|:----------------------|------------------:|:-----------------------|-------------------:|:-----------------------|-------------------:|:----------------------|-------------------:|
|  0 | block_thresholding     | **5.050010768898299** |         0.682723  | 8.219552458397565      |          0.35019   | 8.65583955622039       |          0.197471  | 8.553304771877896     |          0.0732465 |
|  1 | contour_filtering      | 0.7318077210683109    |         0.148523  | 10.700597173253957     |          0.0344229 | 21.318972620240405     |          0.0989541 | 31.035964291263458    |          0.210074  |
|  2 | delaunay_triangulation | 1.505375703066432     |         1.24654   | **15.217867282288136** |          0.208148  | 16.874314375523536     |          4.8556    | 10.847053647410688    |          1.11797   |
|  3 | empty_space            | 2.033170199858218     |         1.52385   | 14.939361117322028     |          0.233282  | **25.021021375929617** |          0.445026  | **34.74279975875022** |          0.478725  |
|  4 | hard_thresholding      | 0.29964070111290103   |         0.0407467 | 10.286864532622754     |          0.0797231 | 20.3181415611083       |          0.0865556 | 30.42901376110176     |          0.0439874 |
### Signal: CosChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_CosChirp.html)    [[Get .csv]](/results/denoising/csv_files/results_CosChirp.csv)
|    | Method + Param         | SNRin=0dB (mean)       |   SNRin=0dB (std) | SNRin=10dB (mean)      |   SNRin=10dB (std) | SNRin=20dB (mean)      |   SNRin=20dB (std) | SNRin=30dB (mean)     |   SNRin=30dB (std) |
|---:|:-----------------------|:-----------------------|------------------:|:-----------------------|-------------------:|:-----------------------|-------------------:|:----------------------|-------------------:|
|  0 | block_thresholding     | 0.04546957708884151    |         0.0793168 | 0.0                    |           0        | 0.0                    |          0         | 0.0                   |           0        |
|  1 | contour_filtering      | 0.801761549076142      |         0.0378754 | 10.719045072506372     |           0.074763 | 21.262116535734418     |          0.158375  | 31.130090680544917    |           0.294472 |
|  2 | delaunay_triangulation | 1.3657790623302517     |         0.89362   | 11.816425296273454     |           1.33392  | 19.63473724453015      |          5.46263   | 9.89719541996494      |           1.00107  |
|  3 | empty_space            | **1.6273235379134576** |         0.933204  | **12.034393918177647** |           0.855093 | **23.820179863418225** |          0.376385  | **32.25149548612126** |           0.22208  |
|  4 | hard_thresholding      | 0.301237960003665      |         0.052625  | 10.27922893860116      |           0.107127 | 20.331222368557334     |          0.0314843 | 30.50931253521395     |           0.142127 |
