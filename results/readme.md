# Benchmark Report

## Configuration   [Get .csv file] 

Length of signals: 512

Repetitions: 100

SNRin values: 
0, 
10, 
20, 
30, 


### Methods  

* hard_thresholding 

### Signals  

* LinearChirp 

* McParallelChirps 

* McParallelChirpsUnbalanced 

## Mean results tables: 
### Signal: LinearChirp  [[View Plot]](plot_LinearChirp.html)    [[Get .csv]](results_LinearChirp.csv)
|    | Method + Param                   |       0 |      10 |      20 |      30 |
|---:|:---------------------------------|--------:|--------:|--------:|--------:|
|  0 | hard_thresholding+{'coeff': 1.0} | 1.34636 | 11.4033 | 21.4487 | 31.4668 |
|  1 | hard_thresholding+{'coeff': 1.5} | 4.04087 | 14.0578 | 23.894  | 33.7937 |
|  2 | hard_thresholding+{'coeff': 2.0} | 6.99755 | 16.5949 | 25.9806 | 35.536  |
|  3 | hard_thresholding+{'coeff': 2.5} | 8.33864 | 17.9087 | 26.9589 | 36.1986 |
|  4 | hard_thresholding+{'coeff': 3.0} | 7.75678 | 18.3765 | 27.297  | 36.4049 |
|  5 | hard_thresholding+{'coeff': 3.5} | 6.0129  | 18.4912 | 27.4539 | 36.4984 |
### Signal: McParallelChirps  [[View Plot]](plot_McParallelChirps.html)    [[Get .csv]](results_McParallelChirps.csv)
|    | Method + Param                   |       0 |      10 |      20 |      30 |
|---:|:---------------------------------|--------:|--------:|--------:|--------:|
|  0 | hard_thresholding+{'coeff': 1.0} | 1.28079 | 11.3131 | 21.3941 | 31.454  |
|  1 | hard_thresholding+{'coeff': 1.5} | 3.63805 | 13.4716 | 23.247  | 33.1427 |
|  2 | hard_thresholding+{'coeff': 2.0} | 5.69718 | 15.2233 | 24.3989 | 33.9404 |
|  3 | hard_thresholding+{'coeff': 2.5} | 5.6488  | 15.9846 | 24.9287 | 34.1514 |
|  4 | hard_thresholding+{'coeff': 3.0} | 4.0005  | 16.1791 | 25.2123 | 34.2181 |
|  5 | hard_thresholding+{'coeff': 3.5} | 2.11538 | 16.0377 | 25.3987 | 34.2578 |
### Signal: McParallelChirpsUnbalanced  [[View Plot]](plot_McParallelChirpsUnbalanced.html)    [[Get .csv]](results_McParallelChirpsUnbalanced.csv)
|    | Method + Param                   |       0 |      10 |      20 |      30 |
|---:|:---------------------------------|--------:|--------:|--------:|--------:|
|  0 | hard_thresholding+{'coeff': 1.0} | 1.28244 | 11.3476 | 21.3741 | 31.4489 |
|  1 | hard_thresholding+{'coeff': 1.5} | 3.68973 | 13.5778 | 23.2308 | 33.1278 |
|  2 | hard_thresholding+{'coeff': 2.0} | 5.7673  | 15.4145 | 24.4169 | 33.9229 |
|  3 | hard_thresholding+{'coeff': 2.5} | 6.00927 | 16.1623 | 24.9231 | 34.1399 |
|  4 | hard_thresholding+{'coeff': 3.0} | 5.04892 | 16.2079 | 25.2011 | 34.222  |
|  5 | hard_thresholding+{'coeff': 3.5} | 3.71074 | 15.7995 | 25.3637 | 34.2984 |
