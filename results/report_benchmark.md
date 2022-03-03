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

* LinearChirp 

* CosChirp 

* McPureTones 

* McCrossingChirps 

* McHarmonic 

* McPureTones 

* McModulatedTones 

* McDoubleCosChirp 

* McSyntheticMixture 

* McSyntheticMixture2 

* HermiteFunction 

* HermiteElipse 

* ToneDumped 

* ToneSharpAttack 

* McOnOffTones 

## Figures:
 ![Summary of results](results_plots.png) 

## Mean results tables: 
### Signal: LinearChirp
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |     6.88761  |      16.3126  |       25.8109 |       35.3893 |
|  1 | delaunay_triangulation |     0.349373 |       7.10522 |       27.1394 |       36.8166 |
|  2 | empty_space            |     4.78402  |      15.1127  |       24.6801 |       34.5569 |
|  3 | hard_thresholding      |     6.78362  |      19.5139  |       28.3413 |       37.5305 |
### Signal: CosChirp
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |     7.03866  |      16.0838  |       25.4208 |       34.9595 |
|  1 | delaunay_triangulation |     0.226723 |       5.43133 |       25.8106 |       35.8028 |
|  2 | empty_space            |     4.40101  |      14.9324  |       24.765  |       34.5426 |
|  3 | hard_thresholding      |     4.80031  |      18.1585  |       27.6963 |       36.9418 |
### Signal: McPureTones
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |   5.72167    |      14.6209  |       23.6746 |       32.9128 |
|  1 | delaunay_triangulation |   0.00705688 |       2.31239 |       22.0046 |       33.1821 |
|  2 | empty_space            |   3.12393    |      14.2294  |       23.7174 |       33.1441 |
|  3 | hard_thresholding      |   0.384915   |      13.9896  |       24.5247 |       33.9608 |
### Signal: McCrossingChirps
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |    5.64264   |      11.9942  |      18.4559  |       31.2424 |
|  1 | delaunay_triangulation |    0.0548535 |       2.29029 |       8.50691 |       10.9209 |
|  2 | empty_space            |    3.57626   |      13.3441  |      21.1832  |       25.5141 |
|  3 | hard_thresholding      |    1.391     |      15.6319  |      25.9126  |       35.2254 |
### Signal: McHarmonic
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |    2.19916   |      4.59861  |       5.07492 |       5.03721 |
|  1 | delaunay_triangulation |   -0.017827  |      0.594128 |       3.80133 |       4.96343 |
|  2 | empty_space            |    2.29446   |     12.6048   |      21.6467  |      27.533   |
|  3 | hard_thresholding      |    0.0547436 |      5.92112  |      17.9113  |      21.2368  |
### Signal: McPureTones
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |   5.72167    |      14.6209  |       23.6746 |       32.9128 |
|  1 | delaunay_triangulation |   0.00705688 |       2.31239 |       22.0046 |       33.1821 |
|  2 | empty_space            |   3.12393    |      14.2294  |       23.7174 |       33.1441 |
|  3 | hard_thresholding      |   0.384915   |      13.9896  |       24.5247 |       33.9608 |
### Signal: McModulatedTones
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |   2.53876    |      13.1281  |       22.6598 |       30.7639 |
|  1 | delaunay_triangulation |   0.00550398 |       1.02655 |        8.8327 |       15.7449 |
|  2 | empty_space            |   2.56635    |      12.9644  |       22.6278 |       29.3566 |
|  3 | hard_thresholding      |   0.136012   |       9.26585 |       20.1317 |       25.5865 |
### Signal: McDoubleCosChirp
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |    6.12832   |      15.1831  |       24.3958 |       33.9541 |
|  1 | delaunay_triangulation |    0.0829883 |       3.24838 |       23.7029 |       34.1717 |
|  2 | empty_space            |    3.67756   |      14.5745  |       24.2495 |       33.8604 |
|  3 | hard_thresholding      |    1.20092   |      15.607   |       25.4595 |       35.0287 |
### Signal: McSyntheticMixture
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |    2.26401   |       5.10163 |       5.41172 |       5.61181 |
|  1 | delaunay_triangulation |   -0.0199624 |       0.5123  |       2.43313 |       3.01095 |
|  2 | empty_space            |    1.40405   |       4.91763 |       7.67549 |       8.17398 |
|  3 | hard_thresholding      |    0.0771595 |       8.08975 |      19.8591  |      23.1746  |
### Signal: McSyntheticMixture2
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |    2.5837    |      9.82945  |      11.3067  |      10.7785  |
|  1 | delaunay_triangulation |   -0.0104577 |      0.475205 |       2.88204 |       4.11329 |
|  2 | empty_space            |    1.91507   |      7.70209  |       9.94653 |      10.1865  |
|  3 | hard_thresholding      |    0.369463  |     12.859    |      23.7072  |      32.9104  |
### Signal: HermiteFunction
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |     8.60162  |      18.3319  |       27.6807 |       37.054  |
|  1 | delaunay_triangulation |     0.291538 |       8.52666 |       29.0863 |       38.548  |
|  2 | empty_space            |     5.02062  |      15.1712  |       25.0827 |       35.0271 |
|  3 | hard_thresholding      |    10.1678   |      20.9788  |       29.9915 |       39.3772 |
### Signal: HermiteElipse
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |     7.13251  |      16.3234  |       25.9015 |       35.691  |
|  1 | delaunay_triangulation |     0.119175 |       3.03886 |       16.6261 |       35.5148 |
|  2 | empty_space            |     4.02571  |      14.7832  |       24.4692 |       32.7403 |
|  3 | hard_thresholding      |     3.87106  |      18.4632  |       27.9214 |       37.1078 |
### Signal: ToneDumped
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |      7.56965 |       17.082  |       26.4192 |       36.1059 |
|  1 | delaunay_triangulation |      2.85049 |       11.9703 |       22.7088 |       24.9411 |
|  2 | empty_space            |      5.18688 |       15.0347 |       24.9076 |       34.9194 |
|  3 | hard_thresholding      |     10.6457  |       19.521  |       28.4307 |       38.2789 |
### Signal: ToneSharpAttack
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |      7.24303 |       14.9448 |       18.386  |       18.4379 |
|  1 | delaunay_triangulation |      1.98792 |       10.7311 |       16.6959 |       26.5657 |
|  2 | empty_space            |      5.00062 |       14.6987 |       24.2626 |       31.1565 |
|  3 | hard_thresholding      |      9.81659 |       17.2565 |       26.856  |       37.1453 |
### Signal: McOnOffTones
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |    4.85342   |       9.55499 |       9.94761 |       17.945  |
|  1 | delaunay_triangulation |    0.0709917 |       3.38287 |      23.9411  |       33.4318 |
|  2 | empty_space            |    3.67338   |      14.4534  |      23.9862  |       33.4736 |
|  3 | hard_thresholding      |    1.06375   |      15.3738  |      25.1974  |       34.4475 |
