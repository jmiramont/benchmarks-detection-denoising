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
### Signal: LinearChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/figures//plot_LinearChirp.html)    [[Get .csv]](results_LinearChirp.csv)
|    | Method + Param         |       0 |       10 |
|---:|:-----------------------|--------:|---------:|
|  0 | block_thresholding     | 5.48839 |  7.87122 |
|  1 | contour_filtering      | 1.96974 | 11.7337  |
|  2 | delaunay_triangulation | 2.84343 | 16.2048  |
|  3 | empty_space            | 3.43143 | 16.2076  |
|  4 | hard_thresholding      | 9.20391 | 18.1778  |
