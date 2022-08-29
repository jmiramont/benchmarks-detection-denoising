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

* ExpChirp 

## Mean results tables: 
### Signal: LinearChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/plot_LinearChirp.html)    [[Get .csv]](results_LinearChirp.csv)
|    | Method + Param         |       0 |      10 |      20 |      30 |
|---:|:-----------------------|--------:|--------:|--------:|--------:|
|  0 | contour_filtering      | 1.9703  | 11.7334 | 22.5629 | 32.3767 |
|  1 | delaunay_triangulation | 2.84343 | 16.2048 | 25.6529 | 35.7068 |
|  2 | empty_space            | 3.43143 | 16.2076 | 25.4908 | 35.9334 |
|  3 | hard_thresholding      | 9.20391 | 18.1778 | 26.8068 | 36.9731 |
### Signal: CosChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/plot_CosChirp.html)    [[Get .csv]](results_CosChirp.csv)
|    | Method + Param         |       0 |      10 |      20 |      30 |
|---:|:-----------------------|--------:|--------:|--------:|--------:|
|  0 | contour_filtering      | 1.66277 | 11.6096 | 22.3803 | 32.2676 |
|  1 | delaunay_triangulation | 1.56267 | 10.9061 | 24.6723 | 34.2547 |
|  2 | empty_space            | 2.1448  | 13.2996 | 24.7019 | 33.5051 |
|  3 | hard_thresholding      | 5.72688 | 17.1129 | 26.5346 | 35.8989 |
### Signal: ExpChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/plot_ExpChirp.html)    [[Get .csv]](results_ExpChirp.csv)
|    | Method + Param         |       0 |      10 |      20 |      30 |
|---:|:-----------------------|--------:|--------:|--------:|--------:|
|  0 | contour_filtering      | 1.6871  | 11.7502 | 22.6876 | 31.6551 |
|  1 | delaunay_triangulation | 2.25985 | 15.6004 | 26.243  | 35.1951 |
|  2 | empty_space            | 2.4676  | 16.0226 | 25.8777 | 35.1853 |
|  3 | hard_thresholding      | 7.70751 | 18.2808 | 27.4144 | 36.5999 |
