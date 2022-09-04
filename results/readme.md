# Benchmark Report

## Configuration   [Get .csv file] 

Length of signals: 512

Repetitions: 50

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
### Signal: LinearChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/figures/html/plot_LinearChirp.html)    [[Get .csv]](./csv_files/results_LinearChirp.csv)
|    | Method + Param         |   SNRin=0dB (mean) |   SNRin=0dB (std) |   SNRin=10dB (mean) |   SNRin=10dB (std) |   SNRin=20dB (mean) |   SNRin=20dB (std) |   SNRin=30dB (mean) |   SNRin=30dB (std) |
|---:|:-----------------------|-------------------:|------------------:|--------------------:|-------------------:|--------------------:|-------------------:|--------------------:|-------------------:|
|  0 | contour_filtering      |            2.18424 |          0.176019 |             12.3336 |           0.205988 |             23.1506 |           0.188686 |             31.8772 |           0.311568 |
|  1 | delaunay_triangulation |            3.24464 |          1.12935  |             17.4635 |           0.754951 |             26.643  |           0.793663 |             31.4023 |           4.94787  |
|  2 | empty_space            |            3.57847 |          1.13441  |             17.0214 |           0.875596 |             26.3286 |           0.726531 |             35.998  |           0.555015 |
|  3 | hard_thresholding      |            6.23946 |          1.26245  |             19.0438 |           0.718967 |             28.2609 |           0.647662 |             37.2749 |           0.531206 |
### Signal: CosChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/figures/html/plot_CosChirp.html)    [[Get .csv]](./csv_files/results_CosChirp.csv)
|    | Method + Param         |   SNRin=0dB (mean) |   SNRin=0dB (std) |   SNRin=10dB (mean) |   SNRin=10dB (std) |   SNRin=20dB (mean) |   SNRin=20dB (std) |   SNRin=30dB (mean) |   SNRin=30dB (std) |
|---:|:-----------------------|-------------------:|------------------:|--------------------:|-------------------:|--------------------:|-------------------:|--------------------:|-------------------:|
|  0 | contour_filtering      |            2.19024 |          0.2524   |             12.3391 |           0.20684  |             23.1562 |           0.245407 |             31.8195 |           0.272845 |
|  1 | delaunay_triangulation |            2.51486 |          0.757284 |             16.7046 |           0.953092 |             26.1832 |           0.798248 |             31.9637 |           5.67399  |
|  2 | empty_space            |            2.85535 |          0.787573 |             16.6151 |           0.670129 |             25.9775 |           0.58175  |             35.3576 |           0.579961 |
|  3 | hard_thresholding      |            4.88581 |          1.02341  |             18.6801 |           0.653845 |             27.9031 |           0.734602 |             36.8783 |           0.531363 |
### Signal: ExpChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/figures/html/plot_ExpChirp.html)    [[Get .csv]](./csv_files/results_ExpChirp.csv)
|    | Method + Param         |   SNRin=0dB (mean) |   SNRin=0dB (std) |   SNRin=10dB (mean) |   SNRin=10dB (std) |   SNRin=20dB (mean) |   SNRin=20dB (std) |   SNRin=30dB (mean) |   SNRin=30dB (std) |
|---:|:-----------------------|-------------------:|------------------:|--------------------:|-------------------:|--------------------:|-------------------:|--------------------:|-------------------:|
|  0 | contour_filtering      |            2.21037 |          0.197814 |             12.3014 |           0.183853 |             23.1751 |           0.22705  |             31.9164 |           0.286927 |
|  1 | delaunay_triangulation |            3.70354 |          1.35119  |             16.878  |           0.97656  |             26.6211 |           0.706433 |             30.2758 |           7.05196  |
|  2 | empty_space            |            4.00188 |          1.29042  |             16.7534 |           0.637199 |             26.4303 |           0.672176 |             36.1347 |           0.534466 |
|  3 | hard_thresholding      |            6.26651 |          1.19612  |             18.8284 |           0.693633 |             28.1422 |           0.56202  |             37.2811 |           0.484208 |
### Signal: ToneSharpAttack  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/figures/html/plot_ToneSharpAttack.html)    [[Get .csv]](./csv_files/results_ToneSharpAttack.csv)
|    | Method + Param         |   SNRin=0dB (mean) |   SNRin=0dB (std) |   SNRin=10dB (mean) |   SNRin=10dB (std) |   SNRin=20dB (mean) |   SNRin=20dB (std) |   SNRin=30dB (mean) |   SNRin=30dB (std) |
|---:|:-----------------------|-------------------:|------------------:|--------------------:|-------------------:|--------------------:|-------------------:|--------------------:|-------------------:|
|  0 | contour_filtering      |            2.19092 |          0.226499 |             12.3166 |           0.28404  |             23.1763 |           0.823508 |             31.5757 |           1.20496  |
|  1 | delaunay_triangulation |            6.81226 |          1.45931  |             14.4248 |           0.852435 |             23.0075 |           1.04288  |             31.4544 |           0.658233 |
|  2 | empty_space            |            6.68377 |          1.10927  |             14.3971 |           0.806682 |             23.735  |           1.08298  |             33.3956 |           0.583782 |
|  3 | hard_thresholding      |            9.63482 |          0.759398 |             16.8454 |           0.579168 |             26.6154 |           0.617364 |             36.6784 |           0.627155 |
### Signal: ToneDumped  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/figures/html/plot_ToneDumped.html)    [[Get .csv]](./csv_files/results_ToneDumped.csv)
|    | Method + Param         |   SNRin=0dB (mean) |   SNRin=0dB (std) |   SNRin=10dB (mean) |   SNRin=10dB (std) |   SNRin=20dB (mean) |   SNRin=20dB (std) |   SNRin=30dB (mean) |   SNRin=30dB (std) |
|---:|:-----------------------|-------------------:|------------------:|--------------------:|-------------------:|--------------------:|-------------------:|--------------------:|-------------------:|
|  0 | contour_filtering      |            2.26616 |          0.259938 |             12.4058 |           0.208591 |             23.4007 |           0.241552 |            31.9574  |           0.256688 |
|  1 | delaunay_triangulation |            7.42676 |          1.63903  |             16.8912 |           1.09633  |             25.2557 |           5.15464  |             6.26128 |           3.37028  |
|  2 | empty_space            |            7.15481 |          1.03136  |             16.6691 |           0.923835 |             26.6247 |           0.925914 |            36.4561  |           0.874759 |
|  3 | hard_thresholding      |           10.2553  |          1.09351  |             19.0934 |           0.831277 |             28.2255 |           0.637572 |            37.878   |           0.633376 |
### Signal: McCrossingChirps  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/figures/html/plot_McCrossingChirps.html)    [[Get .csv]](./csv_files/results_McCrossingChirps.csv)
|    | Method + Param         |   SNRin=0dB (mean) |   SNRin=0dB (std) |   SNRin=10dB (mean) |   SNRin=10dB (std) |   SNRin=20dB (mean) |   SNRin=20dB (std) |   SNRin=30dB (mean) |   SNRin=30dB (std) |
|---:|:-----------------------|-------------------:|------------------:|--------------------:|-------------------:|--------------------:|-------------------:|--------------------:|-------------------:|
|  0 | contour_filtering      |            2.18179 |          0.274746 |            12.2168  |           0.194364 |             22.7892 |           0.202091 |             30.9091 |          0.330529  |
|  1 | delaunay_triangulation |            1.48811 |          0.617819 |             7.77894 |           1.3488   |             10.1999 |           0.772698 |             10.5328 |          0.0536241 |
|  2 | empty_space            |            1.76935 |          0.707629 |             8.74896 |           1.20314  |             10.5995 |           0.108033 |             10.7689 |          0.0117608 |
|  3 | hard_thresholding      |            1.53053 |          0.386528 |            15.521   |           0.575729 |             25.698  |           0.424176 |             34.9242 |          0.419714  |
### Signal: McMultiLinear  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/figures/html/plot_McMultiLinear.html)    [[Get .csv]](./csv_files/results_McMultiLinear.csv)
|    | Method + Param         |   SNRin=0dB (mean) |   SNRin=0dB (std) |   SNRin=10dB (mean) |   SNRin=10dB (std) |   SNRin=20dB (mean) |   SNRin=20dB (std) |   SNRin=30dB (mean) |   SNRin=30dB (std) |
|---:|:-----------------------|-------------------:|------------------:|--------------------:|-------------------:|--------------------:|-------------------:|--------------------:|-------------------:|
|  0 | contour_filtering      |          1.94702   |         0.167461  |            11.4885  |           0.14689  |             21.3673 |           0.184959 |             28.9034 |           0.269683 |
|  1 | delaunay_triangulation |          1.0967    |         0.37384   |            10.5549  |           1.36968  |             23.0375 |           1.89622  |             32.5423 |           0.225487 |
|  2 | empty_space            |          1.14861   |         0.395619  |            12.0785  |           0.783704 |             23.3433 |           0.293575 |             32.74   |           0.250211 |
|  3 | hard_thresholding      |          0.0912397 |         0.0828301 |             7.28721 |           0.562149 |             17.8178 |           0.374289 |             22.1069 |           0.146176 |
### Signal: McCosPlusTone  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/figures/html/plot_McCosPlusTone.html)    [[Get .csv]](./csv_files/results_McCosPlusTone.csv)
|    | Method + Param         |   SNRin=0dB (mean) |   SNRin=0dB (std) |   SNRin=10dB (mean) |   SNRin=10dB (std) |   SNRin=20dB (mean) |   SNRin=20dB (std) |   SNRin=30dB (mean) |   SNRin=30dB (std) |
|---:|:-----------------------|-------------------:|------------------:|--------------------:|-------------------:|--------------------:|-------------------:|--------------------:|-------------------:|
|  0 | contour_filtering      |           2.0963   |          0.155362 |             11.9203 |           0.155682 |             21.8626 |           0.164414 |             29.5247 |           0.32706  |
|  1 | delaunay_triangulation |           1.26364  |          0.422744 |             11.8891 |           1.71805  |             24.1019 |           0.320963 |             31.5897 |           2.49827  |
|  2 | empty_space            |           1.30921  |          0.482095 |             13.3462 |           1.09866  |             24.0792 |           0.317415 |             33.4003 |           0.347112 |
|  3 | hard_thresholding      |           0.254599 |          0.149827 |             11.6249 |           0.546954 |             23.0221 |           0.31186  |             32.4875 |           0.354431 |
### Signal: McMultiCos2  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/figures/html/plot_McMultiCos2.html)    [[Get .csv]](./csv_files/results_McMultiCos2.csv)
|    | Method + Param         |   SNRin=0dB (mean) |   SNRin=0dB (std) |   SNRin=10dB (mean) |   SNRin=10dB (std) |   SNRin=20dB (mean) |   SNRin=20dB (std) |   SNRin=30dB (mean) |   SNRin=30dB (std) |
|---:|:-----------------------|-------------------:|------------------:|--------------------:|-------------------:|--------------------:|-------------------:|--------------------:|-------------------:|
|  0 | contour_filtering      |           2.11959  |          0.254498 |            11.9619  |           0.211527 |             21.8174 |           0.223462 |             28.8865 |           0.303729 |
|  1 | delaunay_triangulation |           0.899634 |          0.377844 |             5.84812 |           0.978    |             11.4599 |           1.31458  |             13.0767 |           1.76318  |
|  2 | empty_space            |           0.922523 |          0.384292 |             8.37601 |           1.0229   |             16.7631 |           0.668108 |             20.0083 |           0.195928 |
|  3 | hard_thresholding      |           0.260505 |          0.185351 |             9.72795 |           0.649537 |             21.4148 |           0.398327 |             28.999  |           0.261845 |
### Signal: McSyntheticMixture2  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/figures/html/plot_McSyntheticMixture2.html)    [[Get .csv]](./csv_files/results_McSyntheticMixture2.csv)
|    | Method + Param         |   SNRin=0dB (mean) |   SNRin=0dB (std) |   SNRin=10dB (mean) |   SNRin=10dB (std) |   SNRin=20dB (mean) |   SNRin=20dB (std) |   SNRin=30dB (mean) |   SNRin=30dB (std) |
|---:|:-----------------------|-------------------:|------------------:|--------------------:|-------------------:|--------------------:|-------------------:|--------------------:|-------------------:|
|  0 | contour_filtering      |           2.18773  |          0.258623 |            12.3241  |           0.180087 |            22.7363  |           0.262196 |            30.3436  |          0.362808  |
|  1 | delaunay_triangulation |           0.537908 |          0.311554 |             3.42473 |           0.515592 |             5.12738 |           0.413437 |             5.24127 |          0.342818  |
|  2 | empty_space            |           0.495393 |          0.365862 |             4.44201 |           0.568102 |             6.20484 |           0.147105 |             6.43008 |          0.0648372 |
|  3 | hard_thresholding      |           0.361867 |          0.190375 |            12.4397  |           0.510207 |            23.467   |           0.353051 |            32.4049  |          0.344349  |
### Signal: HermiteFunction  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/figures/html/plot_HermiteFunction.html)    [[Get .csv]](./csv_files/results_HermiteFunction.csv)
|    | Method + Param         |   SNRin=0dB (mean) |   SNRin=0dB (std) |   SNRin=10dB (mean) |   SNRin=10dB (std) |   SNRin=20dB (mean) |   SNRin=20dB (std) |   SNRin=30dB (mean) |   SNRin=30dB (std) |
|---:|:-----------------------|-------------------:|------------------:|--------------------:|-------------------:|--------------------:|-------------------:|--------------------:|-------------------:|
|  0 | contour_filtering      |            2.27016 |          0.254213 |             12.3274 |           0.205772 |             23.0495 |           0.202082 |             31.6842 |           0.265295 |
|  1 | delaunay_triangulation |            3.1664  |          1.25828  |             17.7518 |           0.996698 |             27.4709 |           0.888298 |             36.9431 |           0.739698 |
|  2 | empty_space            |            3.64155 |          1.46616  |             17.2283 |           0.806352 |             26.9553 |           0.934896 |             36.3979 |           0.636146 |
|  3 | hard_thresholding      |            8.41014 |          1.45319  |             20.3067 |           0.789369 |             29.4855 |           0.841147 |             38.6589 |           0.670954 |
### Signal: McTripleImpulse  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/figures/html/plot_McTripleImpulse.html)    [[Get .csv]](./csv_files/results_McTripleImpulse.csv)
|    | Method + Param         |   SNRin=0dB (mean) |   SNRin=0dB (std) |   SNRin=10dB (mean) |   SNRin=10dB (std) |   SNRin=20dB (mean) |   SNRin=20dB (std) |   SNRin=30dB (mean) |   SNRin=30dB (std) |
|---:|:-----------------------|-------------------:|------------------:|--------------------:|-------------------:|--------------------:|-------------------:|--------------------:|-------------------:|
|  0 | contour_filtering      |            2.20567 |          0.349298 |             10.4449 |           0.76975  |             12.085  |           1.39784  |             11.0325 |           1.18127  |
|  1 | delaunay_triangulation |            2.06383 |          0.576965 |             13.0139 |           1.25654  |             23.8194 |           0.654943 |             33.6128 |           0.383849 |
|  2 | empty_space            |            2.22948 |          0.705826 |             13.5938 |           0.715925 |             24.169  |           0.473401 |             33.9134 |           0.362491 |
|  3 | hard_thresholding      |            5.44254 |          0.838539 |             16.6416 |           0.624696 |             26.2946 |           0.515767 |             35.9259 |           0.438907 |
### Signal: McOnOffTones  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/figures/html/plot_McOnOffTones.html)    [[Get .csv]](./csv_files/results_McOnOffTones.csv)
|    | Method + Param         |   SNRin=0dB (mean) |   SNRin=0dB (std) |   SNRin=10dB (mean) |   SNRin=10dB (std) |   SNRin=20dB (mean) |   SNRin=20dB (std) |   SNRin=30dB (mean) |   SNRin=30dB (std) |
|---:|:-----------------------|-------------------:|------------------:|--------------------:|-------------------:|--------------------:|-------------------:|--------------------:|-------------------:|
|  0 | contour_filtering      |           2.17882  |          0.196763 |             12.1334 |           0.226738 |             22.2626 |           0.218647 |             29.532  |           0.319716 |
|  1 | delaunay_triangulation |           2.06274  |          0.750042 |             14.503  |           2.42973  |             24.4344 |           0.412449 |             31.0556 |           4.80153  |
|  2 | empty_space            |           2.23624  |          0.886553 |             15.4942 |           0.608907 |             24.6131 |           0.367915 |             33.5549 |           0.322655 |
|  3 | hard_thresholding      |           0.969269 |          0.377888 |             15.0802 |           0.571964 |             25.0696 |           0.465125 |             34.0884 |           0.345108 |
### Signal: McOnOff2  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/figures/html/plot_McOnOff2.html)    [[Get .csv]](./csv_files/results_McOnOff2.csv)
|    | Method + Param         |   SNRin=0dB (mean) |   SNRin=0dB (std) |   SNRin=10dB (mean) |   SNRin=10dB (std) |   SNRin=20dB (mean) |   SNRin=20dB (std) |   SNRin=30dB (mean) |   SNRin=30dB (std) |
|---:|:-----------------------|-------------------:|------------------:|--------------------:|-------------------:|--------------------:|-------------------:|--------------------:|-------------------:|
|  0 | contour_filtering      |           2.12041  |          0.176303 |             11.9915 |           0.182418 |             22.1691 |           0.25947  |             30.2742 |           0.290857 |
|  1 | delaunay_triangulation |           1.90587  |          0.740903 |             14.7415 |           1.49057  |             23.5631 |           3.24074  |             31.3041 |           4.109    |
|  2 | empty_space            |           2.08954  |          0.791938 |             15.2769 |           0.689622 |             24.4639 |           0.430964 |             33.6198 |           0.346141 |
|  3 | hard_thresholding      |           0.815486 |          0.326004 |             14.5503 |           0.555708 |             24.8077 |           0.452955 |             34.0961 |           0.366519 |
