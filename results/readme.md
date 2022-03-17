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

## Figures:
 ![Summary of results](results/../figures/plots_grid.png) 

## Mean results tables: 
### Signal: CosChirp
|    | Method + Param         |   SNRin: 0dB |   SNRin: 10dB |   SNRin: 20dB |   SNRin: 30dB |
|---:|:-----------------------|-------------:|--------------:|--------------:|--------------:|
|  0 | contour_filtering      |      2.05426 |       12.0966 |       23.0161 |       32.4539 |
|  1 | delaunay_triangulation |      0.97065 |       12.4395 |       26.3115 |       35.8374 |
|  2 | empty_space            |      2.26223 |       16.3672 |       26.39   |       35.8414 |
|  3 | hard_thresholding      |      4.82755 |       18.2149 |       27.7513 |       36.9504 |
