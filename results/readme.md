# Benchmark Report

## Configuration   [Get .csv file] 

Length of signals: 512

Repetitions: 5

SNRin values: 
0, 
10, 


### Methods  

* contour_filtering 

* delaunay_triangulation 

* empty_space 

* hard_thresholding 

### Signals  

* LinearChirp 

## Mean results tables: 
### Signal: LinearChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/plot_LinearChirp.html)    [[Get .csv]](results_LinearChirp.csv)
|    | Method + Param         |       0 |      10 |
|---:|:-----------------------|--------:|--------:|
|  0 | contour_filtering      | 1.9703  | 11.7334 |
|  1 | delaunay_triangulation | 2.84343 | 16.2048 |
|  2 | empty_space            | 3.43143 | 16.2076 |
|  3 | hard_thresholding      | 9.20391 | 18.1778 |
