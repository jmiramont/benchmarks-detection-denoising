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
|  0 | contour_filtering      |     5.37611  |      14.8291  |       24.3335 |       34.0039 |
|  1 | delaunay_triangulation |     0.194121 |       6.21441 |       26.4098 |       34.6288 |
|  2 | empty_space            |     4.63164  |      15.2044  |       24.8991 |       34.3039 |
|  3 | hard_thresholding      |     3.67097  |      17.6747  |       27.347  |       36.2795 |
### Signal: CosChirp
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |    5.5957    |      14.7418  |       24.1067 |       33.4723 |
|  1 | delaunay_triangulation |    0.0879361 |       3.67727 |       24.6274 |       34.2047 |
|  2 | empty_space            |    3.76604   |      14.9898  |       24.4756 |       34.195  |
|  3 | hard_thresholding      |    2.20706   |      16.6787  |       26.2546 |       35.514  |
### Signal: McPureTones
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |    3.87527   |      12.9026  |       21.9228 |       30.0082 |
|  1 | delaunay_triangulation |   -0.0233088 |       1.22631 |       12.2098 |       26.0067 |
|  2 | empty_space            |    2.50962   |      13.5099  |       23.0143 |       31.2732 |
|  3 | hard_thresholding      |    0.0865319 |       9.63334 |       21.651  |       28.3862 |
### Signal: McCrossingChirps
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |   4.35822    |      11.4017  |      21.1465  |      26.2595  |
|  1 | delaunay_triangulation |   0.00563507 |       1.66056 |       5.68825 |       7.61849 |
|  2 | empty_space            |   2.54744    |      10.7743  |      15.9499  |      21.3545  |
|  3 | hard_thresholding      |   0.543605   |      13.087   |      24.6301  |      33.9669  |
### Signal: McHarmonic
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |    3.99975   |      12.7871  |      18.6917  |      20.7394  |
|  1 | delaunay_triangulation |   -0.0155313 |       1.04961 |       5.65763 |       6.98011 |
|  2 | empty_space            |    2.83425   |      13.4911  |      22.5239  |      28.2773  |
|  3 | hard_thresholding      |    0.135767  |       9.36493 |      20.4358  |      27.5871  |
### Signal: McPureTones
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |    3.87527   |      12.9026  |       21.9228 |       30.0082 |
|  1 | delaunay_triangulation |   -0.0233088 |       1.22631 |       12.2098 |       26.0067 |
|  2 | empty_space            |    2.50962   |      13.5099  |       23.0143 |       31.2732 |
|  3 | hard_thresholding      |    0.0865319 |       9.63334 |       21.651  |       28.3862 |
