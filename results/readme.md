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
|  0 | contour_filtering      |     2.05426  |       12.0966 |       23.0161 |       32.4539 |
|  1 | delaunay_triangulation |     0.991067 |       10.599  |       26.1649 |       34.1985 |
|  2 | empty_space            |     2.0862   |       12.6876 |       21.3135 |       24.2428 |
|  3 | hard_thresholding      |     4.82755  |       18.2149 |       27.7513 |       36.9504 |
### Signal: ExpChirp
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |      1.99722 |       12.0325 |       23.1567 |       32.5162 |
|  1 | delaunay_triangulation |      2.00249 |       16.4761 |       27.4758 |       32.265  |
|  2 | empty_space            |      3.27627 |       15.6651 |       25.606  |       34.8977 |
|  3 | hard_thresholding      |      6.4353  |       19.3687 |       28.7225 |       37.7722 |
### Signal: ToneSharpAttack
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |      2.00751 |       12.044  |       23.135  |       33.1745 |
|  1 | delaunay_triangulation |      5.99181 |       13.8755 |       20.7711 |       31.3556 |
|  2 | empty_space            |      6.94742 |       14.3341 |       21.4909 |       27.7909 |
|  3 | hard_thresholding      |      9.79415 |       16.9689 |       27.1718 |       36.9764 |
### Signal: McCrossingChirps
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |     1.99648  |      12.1061  |       22.6757 |       31.87   |
|  1 | delaunay_triangulation |     0.799922 |       7.95157 |       19.8556 |       33.2495 |
|  2 | empty_space            |     1.60693  |       8.98514 |       13.2415 |       13.5748 |
|  3 | hard_thresholding      |     1.91306  |      16.1181  |       25.9742 |       35.2584 |
### Signal: McMultiLinear
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |    1.71799   |      11.452   |       21.8017 |       30.0746 |
|  1 | delaunay_triangulation |    0.398671  |       6.74071 |       23.4867 |       32.5293 |
|  2 | empty_space            |    1.04752   |       9.15715 |       20.8292 |       27.8551 |
|  3 | hard_thresholding      |    0.0578374 |       6.72027 |       18.6306 |       22.673  |
### Signal: McCosPlusTone
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |     1.92458  |      11.8641  |       22.0646 |       30.6064 |
|  1 | delaunay_triangulation |     0.510943 |       8.28006 |       24.4633 |       33.3854 |
|  2 | empty_space            |     1.26012  |      10.0501  |       21.6921 |       30.9849 |
|  3 | hard_thresholding      |     0.232175 |      12.1335  |       23.3928 |       32.7175 |
### Signal: McMultiCos
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |     1.98898  |      11.8785  |      22.0073  |      30.5597  |
|  1 | delaunay_triangulation |     0.249946 |       3.43129 |       6.83161 |       6.64561 |
|  2 | empty_space            |     0.842131 |       5.26183 |       8.53568 |       8.84006 |
|  3 | hard_thresholding      |     0.155871 |       9.27576 |      20.2919  |      25.6532  |
### Signal: McSyntheticMixture2
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |     2.00869  |      12.205   |      22.8953  |      31.5994  |
|  1 | delaunay_triangulation |     0.304352 |       1.94917 |       3.87547 |       4.44011 |
|  2 | empty_space            |     0.744628 |       2.83118 |       3.91917 |       4.10255 |
|  3 | hard_thresholding      |     0.4018   |      12.8998  |      23.7702  |      32.7359  |
### Signal: McSyntheticMixture3
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |      1.95898 |       12.0101 |       22.8372 |       32.1322 |
|  1 | delaunay_triangulation |      1.48621 |       14.8343 |       26.4426 |       27.9359 |
|  2 | empty_space            |      2.82413 |       14.393  |       24.4192 |       32.9763 |
|  3 | hard_thresholding      |      4.74571 |       18.3321 |       27.5506 |       36.6471 |
### Signal: HermiteFunction
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |      2.11764 |       12.12   |       23.1645 |       32.5352 |
|  1 | delaunay_triangulation |      1.5682  |       16.3722 |       28.4414 |       37.7757 |
|  2 | empty_space            |      3.01573 |       15.5286 |       23.5574 |       27.4866 |
|  3 | hard_thresholding      |     10.2413  |       21.0202 |       30.0284 |       39.4325 |
### Signal: McImpulses
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |     1.94294  |       9.22026 |       9.27883 |       8.09881 |
|  1 | delaunay_triangulation |     0.566082 |       4.54296 |      10.9548  |      19.6005  |
|  2 | empty_space            |     1.28384  |       6.70706 |      13.5853  |      16.0208  |
|  3 | hard_thresholding      |     2.74866  |      11.0686  |      19.4825  |      27.5936  |
### Signal: McTripleImpulse
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |      2.26497 |       11.1382 |       13.3505 |       12.3139 |
|  1 | delaunay_triangulation |      1.00948 |       10.4801 |       22.6034 |       33.8971 |
|  2 | empty_space            |      2.07139 |       11.76   |       22.0344 |       31.9016 |
|  3 | hard_thresholding      |      5.83679 |       16.9719 |       26.6062 |       36.3027 |
