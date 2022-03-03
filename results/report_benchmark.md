# Benchmark Report 

## Configuration 

Length of signals: 256

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

## Figures:
 ![Summary of results](results_plots.png) 

## Mean results tables: 
### Signal: LinearChirp
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |      5.25526 |       14.8983 |       24.3755 |       34.2228 |
|  1 | delaunay_triangulation |      0.2365  |        6.4637 |       26.2923 |       34.4135 |
|  2 | empty_space            |      4.50075 |       15.3304 |       24.8752 |       34.5495 |
|  3 | hard_thresholding      |      3.45633 |       18.1912 |       27.1993 |       36.5214 |
### Signal: CosChirp
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |     5.40821  |      14.6312  |       23.9438 |       33.3107 |
|  1 | delaunay_triangulation |     0.135958 |       3.84224 |       25.1173 |       34.2644 |
|  2 | empty_space            |     4.02329  |      14.9703  |       24.4234 |       34.097  |
|  3 | hard_thresholding      |     2.16104  |      16.554   |       26.2753 |       35.4414 |
### Signal: McPureTones
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |    3.63424   |      12.9234  |       21.9137 |       30.1896 |
|  1 | delaunay_triangulation |   -0.0218288 |       1.10634 |       11.038  |       26.1393 |
|  2 | empty_space            |    2.50488   |      13.5484  |       23.0763 |       31.5372 |
|  3 | hard_thresholding      |    0.0716206 |       9.52896 |       21.7404 |       28.6445 |
### Signal: McCrossingChirps
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |   4.60322    |      11.6056  |      20.6873  |      26.1638  |
|  1 | delaunay_triangulation |  -0.00915737 |       1.57517 |       5.64882 |       7.53595 |
|  2 | empty_space            |   2.75053    |      11.0876  |      16.4024  |      19.2933  |
|  3 | hard_thresholding      |   0.576684   |      13.1484  |      24.5467  |      33.9678  |
### Signal: McHarmonic
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |    4.28066   |      12.7663  |      18.6313  |      20.5168  |
|  1 | delaunay_triangulation |   -0.0384117 |       1.21922 |       5.71179 |       7.05813 |
|  2 | empty_space            |    2.95065   |      13.5437  |      22.4804  |      28.2918  |
|  3 | hard_thresholding      |    0.109989  |       9.47571 |      20.3174  |      27.4824  |
### Signal: McPureTones
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |    3.63424   |      12.9234  |       21.9137 |       30.1896 |
|  1 | delaunay_triangulation |   -0.0218288 |       1.10634 |       11.038  |       26.1393 |
|  2 | empty_space            |    2.50488   |      13.5484  |       23.0763 |       31.5372 |
|  3 | hard_thresholding      |    0.0716206 |       9.52896 |       21.7404 |       28.6445 |
