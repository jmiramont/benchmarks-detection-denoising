# Benchmark Report

## Configuration   [Get .csv file] 

Length of signals: 1024

Repetitions: 100

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

* ToneSharpAttack 

* ToneDumped 

* McCrossingChirps 

* McMultiLinear 

* McCosPlusTone 

* McMultiCos2 

* McSyntheticMixture2 

* HermiteFunction 

* McTripleImpulse 

* McOnOffTones 

* McOnOff2 

## Mean results tables: 
### Signal: LinearChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/plot_LinearChirp.html)    [[Get .csv]](results_LinearChirp.csv)
|    | Method + Param         |        0 |      10 |      20 |      30 |
|---:|:-----------------------|---------:|--------:|--------:|--------:|
|  0 | contour_filtering      |  1.5517  | 11.6568 | 22.7981 | 32.4491 |
|  1 | delaunay_triangulation |  4.02977 | 16.8417 | 26.7481 | 36.419  |
|  2 | empty_space            |  4.56396 | 16.8064 | 26.5275 | 36.3102 |
|  3 | hard_thresholding      | 10.3558  | 19.7837 | 28.7499 | 38.0833 |
### Signal: CosChirp  [[View Plot]](../gh-pages/results/plot_CosChirp.html)    [[Get .csv]](results_CosChirp.csv)
|    | Method + Param         |       0 |      10 |      20 |      30 |
|---:|:-----------------------|--------:|--------:|--------:|--------:|
|  0 | contour_filtering      | 1.53124 | 11.5898 | 22.618  | 32.2653 |
|  1 | delaunay_triangulation | 2.32484 | 15.3947 | 25.8912 | 34.205  |
|  2 | empty_space            | 2.7412  | 16.0616 | 25.812  | 34.5548 |
|  3 | hard_thresholding      | 8.53445 | 18.7899 | 27.8157 | 36.966  |
### Signal: ExpChirp  [[View Plot]](../gh-pages/results/plot_ExpChirp.html)    [[Get .csv]](results_ExpChirp.csv)
|    | Method + Param         |       0 |      10 |      20 |      30 |
|---:|:-----------------------|--------:|--------:|--------:|--------:|
|  0 | contour_filtering      | 1.5705  | 11.6677 | 22.7937 | 32.3298 |
|  1 | delaunay_triangulation | 3.79558 | 16.9493 | 26.749  | 28.9912 |
|  2 | empty_space            | 4.32234 | 16.8585 | 26.6328 | 36.2094 |
|  3 | hard_thresholding      | 9.80709 | 19.5064 | 28.7002 | 37.8922 |
### Signal: ToneSharpAttack  [[View Plot]](../gh-pages/results/plot_ToneSharpAttack.html)    [[Get .csv]](results_ToneSharpAttack.csv)
|    | Method + Param         |        0 |      10 |      20 |      30 |
|---:|:-----------------------|---------:|--------:|--------:|--------:|
|  0 | contour_filtering      |  1.5318  | 11.5909 | 22.5855 | 32.6084 |
|  1 | delaunay_triangulation |  6.53185 | 14.6663 | 22.9683 | 33.0777 |
|  2 | empty_space            |  6.65567 | 15.0537 | 23.5418 | 34.3087 |
|  3 | hard_thresholding      | 10.8136  | 18.2668 | 27.7207 | 37.397  |
### Signal: ToneDumped  [[View Plot]](../gh-pages/results/plot_ToneDumped.html)    [[Get .csv]](results_ToneDumped.csv)
|    | Method + Param         |        0 |      10 |      20 |      30 |
|---:|:-----------------------|---------:|--------:|--------:|--------:|
|  0 | contour_filtering      |  1.57943 | 11.6787 | 22.9724 | 32.6222 |
|  1 | delaunay_triangulation |  7.02725 | 17.5939 | 27.8615 | 21.759  |
|  2 | empty_space            |  7.08354 | 17.5555 | 27.681  | 37.3164 |
|  3 | hard_thresholding      | 11.1368  | 20.6796 | 29.9686 | 39.1794 |
### Signal: McCrossingChirps  [[View Plot]](../gh-pages/results/plot_McCrossingChirps.html)    [[Get .csv]](results_McCrossingChirps.csv)
|    | Method + Param         |       0 |      10 |      20 |      30 |
|---:|:-----------------------|--------:|--------:|--------:|--------:|
|  0 | contour_filtering      | 1.52382 | 11.6076 | 22.5934 | 31.9694 |
|  1 | delaunay_triangulation | 1.97401 | 13.123  | 22.5374 | 34.0291 |
|  2 | empty_space            | 2.07698 | 14.1535 | 24.436  | 30.7966 |
|  3 | hard_thresholding      | 6.42958 | 17.2067 | 26.463  | 35.6593 |
### Signal: McMultiLinear  [[View Plot]](../gh-pages/results/plot_McMultiLinear.html)    [[Get .csv]](results_McMultiLinear.csv)
|    | Method + Param         |       0 |      10 |      20 |      30 |
|---:|:-----------------------|--------:|--------:|--------:|--------:|
|  0 | contour_filtering      | 1.58318 | 11.5979 | 21.9222 | 30.558  |
|  1 | delaunay_triangulation | 1.59098 | 13.5353 | 24.5905 | 33.8343 |
|  2 | empty_space            | 1.8482  | 14.6119 | 24.6437 | 33.8771 |
|  3 | hard_thresholding      | 2.78419 | 14.9438 | 24.2048 | 33.2693 |
### Signal: McCosPlusTone  [[View Plot]](../gh-pages/results/plot_McCosPlusTone.html)    [[Get .csv]](results_McCosPlusTone.csv)
|    | Method + Param         |       0 |      10 |      20 |      30 |
|---:|:-----------------------|--------:|--------:|--------:|--------:|
|  0 | contour_filtering      | 1.57935 | 11.6104 | 22.1602 | 30.9971 |
|  1 | delaunay_triangulation | 1.47696 | 12.2408 | 24.5769 | 33.4238 |
|  2 | empty_space            | 1.7834  | 13.9689 | 24.5822 | 33.89   |
|  3 | hard_thresholding      | 3.29994 | 15.766  | 24.9105 | 34.0387 |
### Signal: McMultiCos2  [[View Plot]](../gh-pages/results/plot_McMultiCos2.html)    [[Get .csv]](results_McMultiCos2.csv)
|    | Method + Param         |       0 |       10 |       20 |       30 |
|---:|:-----------------------|--------:|---------:|---------:|---------:|
|  0 | contour_filtering      | 1.48446 | 11.5742  | 21.9589  | 30.6764  |
|  1 | delaunay_triangulation | 1.20036 |  6.79344 |  8.88913 |  9.18267 |
|  2 | empty_space            | 1.33433 |  8.6872  | 11.044   | 11.3649  |
|  3 | hard_thresholding      | 2.5126  | 14.5757  | 23.6265  | 32.7042  |
### Signal: McSyntheticMixture2  [[View Plot]](../gh-pages/results/plot_McSyntheticMixture2.html)    [[Get .csv]](results_McSyntheticMixture2.csv)
|    | Method + Param         |        0 |       10 |       20 |       30 |
|---:|:-----------------------|---------:|---------:|---------:|---------:|
|  0 | contour_filtering      | 1.47686  | 11.6328  | 22.4382  | 31.2383  |
|  1 | delaunay_triangulation | 0.699509 |  2.51569 |  3.27822 |  3.50459 |
|  2 | empty_space            | 0.699529 |  2.80001 |  3.59659 |  3.66524 |
|  3 | hard_thresholding      | 2.88143  | 14.8787  | 24.0024  | 33.1886  |
### Signal: HermiteFunction  [[View Plot]](../gh-pages/results/plot_HermiteFunction.html)    [[Get .csv]](results_HermiteFunction.csv)
|    | Method + Param         |        0 |      10 |      20 |      30 |
|---:|:-----------------------|---------:|--------:|--------:|--------:|
|  0 | contour_filtering      |  1.46426 | 11.3148 | 22.1428 | 29.8105 |
|  1 | delaunay_triangulation |  3.7397  | 17.6714 | 27.3131 | 36.8352 |
|  2 | empty_space            |  4.27106 | 17.402  | 27.095  | 36.6831 |
|  3 | hard_thresholding      | 10.7503  | 20.2558 | 29.3087 | 38.5736 |
### Signal: McTripleImpulse  [[View Plot]](../gh-pages/results/plot_McTripleImpulse.html)    [[Get .csv]](results_McTripleImpulse.csv)
|    | Method + Param         |       0 |      10 |      20 |      30 |
|---:|:-----------------------|--------:|--------:|--------:|--------:|
|  0 | contour_filtering      | 1.41429 | 11.4086 | 19.7015 | 24.8857 |
|  1 | delaunay_triangulation | 1.52195 | 11.4161 | 22.5564 | 32.6803 |
|  2 | empty_space            | 1.63118 | 12.3395 | 23.0552 | 33.0642 |
|  3 | hard_thresholding      | 6.17018 | 16.3369 | 25.5547 | 34.813  |
### Signal: McOnOffTones  [[View Plot]](../gh-pages/results/plot_McOnOffTones.html)    [[Get .csv]](results_McOnOffTones.csv)
|    | Method + Param         |       0 |      10 |      20 |      30 |
|---:|:-----------------------|--------:|--------:|--------:|--------:|
|  0 | contour_filtering      | 1.59047 | 11.6647 | 22.4126 | 31.4796 |
|  1 | delaunay_triangulation | 2.9628  | 16.0261 | 25.86   | 34.906  |
|  2 | empty_space            | 3.35261 | 16.3324 | 25.8115 | 35.0362 |
|  3 | hard_thresholding      | 5.78558 | 16.813  | 25.8962 | 34.8893 |
### Signal: McOnOff2  [[View Plot]](../gh-pages/results/plot_McOnOff2.html)    [[Get .csv]](results_McOnOff2.csv)
|    | Method + Param         |       0 |      10 |      20 |      30 |
|---:|:-----------------------|--------:|--------:|--------:|--------:|
|  0 | contour_filtering      | 1.57572 | 11.6565 | 22.402  | 31.3705 |
|  1 | delaunay_triangulation | 2.69419 | 15.7531 | 25.5637 | 34.2289 |
|  2 | empty_space            | 3.17603 | 16.1806 | 25.6617 | 34.8841 |
|  3 | hard_thresholding      | 5.64481 | 16.7177 | 25.8267 | 34.8678 |
