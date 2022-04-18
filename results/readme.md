# Benchmark Report 

## Configuration 

Length of signals: 512

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

* identity 

### Signals  

* CosChirp 

* ExpChirp 

* ToneSharpAttack 

* McCrossingChirps 

* McMultiLinear 

* McCosPlusTone 

* McMultiCos 

* McSyntheticMixture2 

* McSyntheticMixture3 

* HermiteFunction 

* McImpulses 

* McTripleImpulse 

## Figures:
 ![Summary of results](results/../figures/plots_grid.png) 

## Mean results tables: 
### Signal: CosChirp
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |  2.05426     |       12.0966 |       23.0161 |       32.4539 |
|  1 | delaunay_triangulation |  0.97065     |       12.4395 |       26.3115 |       35.8374 |
|  2 | empty_space            |  2.26223     |       16.3672 |       26.39   |       35.8414 |
|  3 | hard_thresholding      |  4.82755     |       18.2149 |       27.7513 |       36.9504 |
|  4 | identity               |  4.33947e-16 |       10      |       20      |       30      |
### Signal: ExpChirp
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |  1.99722     |       12.0325 |       23.1567 |       32.5162 |
|  1 | delaunay_triangulation |  1.56534     |       15.5307 |       27.2599 |       36.619  |
|  2 | empty_space            |  3.1355      |       17.1141 |       27.1871 |       36.5277 |
|  3 | hard_thresholding      |  6.4353      |       19.3687 |       28.7225 |       37.7722 |
|  4 | identity               |  3.21442e-17 |       10      |       20      |       30      |
### Signal: ToneSharpAttack
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |  2.00751     |      12.044   |       23.135  |       33.1745 |
|  1 | delaunay_triangulation |  4.21406     |      13.6331  |       20.6572 |       30.3836 |
|  2 | empty_space            |  6.45899     |      14.9482  |       23.2178 |       31.2125 |
|  3 | hard_thresholding      |  9.79415     |      16.9689  |       27.1718 |       36.9764 |
|  4 | identity               | -1.02862e-15 |       9.99941 |       19.9935 |       29.9349 |
### Signal: McCrossingChirps
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |  1.99648     |      12.1061  |       22.6757 |       31.87   |
|  1 | delaunay_triangulation |  0.613102    |       6.14106 |       13.3697 |       18.7615 |
|  2 | empty_space            |  1.61229     |       9.90277 |       16.4416 |       21.0859 |
|  3 | hard_thresholding      |  1.91306     |      16.1181  |       25.9742 |       35.2584 |
|  4 | identity               |  5.14308e-16 |      10       |       20      |       30      |
### Signal: McMultiLinear
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |  1.71799     |      11.452   |      21.8017  |      30.0746  |
|  1 | delaunay_triangulation |  0.215266    |       3.40387 |       7.84893 |       7.75055 |
|  2 | empty_space            |  0.769624    |       6.27851 |       9.20819 |       8.74424 |
|  3 | hard_thresholding      |  0.0578374   |       6.72027 |      18.6306  |      22.673   |
|  4 | identity               |  8.67895e-16 |      10       |      20       |      30       |
### Signal: McCosPlusTone
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |  1.92458     |      11.8641  |       22.0646 |       30.6064 |
|  1 | delaunay_triangulation |  0.403829    |       6.44689 |       23.3764 |       32.8224 |
|  2 | empty_space            |  1.16288     |      10.8346  |       23.6571 |       32.9322 |
|  3 | hard_thresholding      |  0.232175    |      12.1335  |       23.3928 |       32.7175 |
|  4 | identity               |  1.02862e-15 |      10       |       20      |       30      |
### Signal: McMultiCos
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |   1.98898    |      11.8785  |       22.0073 |       30.5597 |
|  1 | delaunay_triangulation |   0.328075   |       3.87337 |       14.2497 |       15.3449 |
|  2 | empty_space            |   1.0058     |       7.68371 |       16.9725 |       19.6562 |
|  3 | hard_thresholding      |   0.155871   |       9.27576 |       20.2919 |       25.6532 |
|  4 | identity               |  -5.3038e-16 |      10       |       20      |       30      |
### Signal: McSyntheticMixture2
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |   2.00869    |      12.205   |      22.8953  |      31.5994  |
|  1 | delaunay_triangulation |   0.181568   |       2.08626 |       5.15693 |       6.43157 |
|  2 | empty_space            |   0.685705   |       3.8831  |       6.50632 |       7.21349 |
|  3 | hard_thresholding      |   0.4018     |      12.8998  |      23.7702  |      32.7359  |
|  4 | identity               |   2.2501e-16 |      10       |      20       |      30       |
### Signal: McSyntheticMixture3
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |  1.95898     |       12.0101 |       22.8372 |       32.1322 |
|  1 | delaunay_triangulation |  1.31958     |       14.0444 |       26.079  |       25.9748 |
|  2 | empty_space            |  2.81197     |       16.7143 |       26.4293 |       35.6425 |
|  3 | hard_thresholding      |  4.74571     |       18.3321 |       27.5506 |       36.6471 |
|  4 | identity               | -2.73226e-16 |       10      |       20      |       30      |
### Signal: HermiteFunction
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |  2.11764     |       12.12   |       23.1645 |       32.5352 |
|  1 | delaunay_triangulation |  1.90213     |       17.8815 |       28.4989 |       37.9971 |
|  2 | empty_space            |  3.76274     |       18.2889 |       28.0573 |       37.4906 |
|  3 | hard_thresholding      | 10.2413      |       21.0202 |       30.0284 |       39.4325 |
|  4 | identity               | -2.73226e-16 |       10      |       20      |       30      |
### Signal: McImpulses
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |  1.94294     |       9.22026 |       9.27883 |       8.09881 |
|  1 | delaunay_triangulation |  1.02554     |       8.75948 |      20.081   |      31.206   |
|  2 | empty_space            |  2.23856     |      11.3448  |      21.9631  |      32.3536  |
|  3 | hard_thresholding      |  2.74866     |      11.0686  |      19.4825  |      27.5936  |
|  4 | identity               |  4.01803e-16 |       9.99995 |      19.9995  |      29.9946  |
### Signal: McTripleImpulse
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |  2.26497     |      11.1382  |       13.3505 |       12.3139 |
|  1 | delaunay_triangulation |  1.84126     |      13.8204  |       25.4992 |       34.267  |
|  2 | empty_space            |  3.48872     |      15.4611  |       25.8402 |       35.242  |
|  3 | hard_thresholding      |  5.83679     |      16.9719  |       26.6062 |       36.3027 |
|  4 | identity               | -2.73226e-16 |       9.99997 |       19.9997 |       29.9971 |
