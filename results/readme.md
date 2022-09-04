# Benchmark Report

## Configuration   [Get .csv file] 

Length of signals: 512

Repetitions: 5

SNRin values: 
0, 
10, 


### Methods  

* block_thresholding 

* contour_filtering 

* delaunay_triangulation 

* empty_space 

* hard_thresholding 

### Signals  

* LinearChirp 

## Mean results tables: 
### Signal: LinearChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/figures/html/plot_LinearChirp.html)    [[Get .csv]](./csv_files/results_LinearChirp.csv)
|    | Method + Param         |   SNRin=0dB (mean) |   SNRin=0dB (std) |   SNRin=10dB (mean) |   SNRin=10dB (std) |
|---:|:-----------------------|-------------------:|------------------:|--------------------:|-------------------:|
|  0 | block_thresholding     |            5.48839 |          0.481672 |             7.87122 |           0.640105 |
|  1 | contour_filtering      |            1.96974 |          0.270486 |            11.7337  |           0.167603 |
|  2 | delaunay_triangulation |            2.84343 |          1.00184  |            16.2048  |           0.409093 |
|  3 | empty_space            |            3.43143 |          1.28961  |            16.2076  |           0.590756 |
|  4 | hard_thresholding      |            9.20391 |          0.678006 |            18.1778  |           0.504933 |
