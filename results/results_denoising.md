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

Results shown here are the mean and standard deviation of                               the Quality Reconstruction Factor.                               Best performances are **bolded**. 
### Signal: LinearChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_LinearChirp.html)    [[Get .csv]](.\denoising\csv_files\results_LinearChirp.csv)
|    | Method + Param         | SNRin=0dB (mean)     |   SNRin=0dB (std) | SNRin=10dB (mean)      |   SNRin=10dB (std) | SNRin=20dB (mean)     |   SNRin=20dB (std) | SNRin=30dB (mean)      |   SNRin=30dB (std) |
|---:|:-----------------------|:---------------------|------------------:|:-----------------------|-------------------:|:----------------------|-------------------:|:-----------------------|-------------------:|
|  0 | contour_filtering      | 2.1842383566076364   |          0.176019 | 12.333599016007968     |           0.205988 | 23.150569453465895    |           0.188686 | 31.87718822453381      |           0.311568 |
|  1 | delaunay_triangulation | 3.244635734908947    |          1.12935  | 17.463496994434678     |           0.754951 | 26.64300331484412     |           0.793663 | 31.402327992477645     |           4.94787  |
|  2 | empty_space            | 3.578466625046388    |          1.13441  | 17.021398596985968     |           0.875596 | 26.328558935689788    |           0.726531 | 35.99802379110711      |           0.555015 |
|  3 | hard_thresholding      | **6.23945563985175** |          1.26245  | **19.043809829433773** |           0.718967 | **28.26091397202167** |           0.647662 | **37.274938085932106** |           0.531206 |
### Signal: CosChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_CosChirp.html)    [[Get .csv]](.\denoising\csv_files\results_CosChirp.csv)
|    | Method + Param         | SNRin=0dB (mean)      |   SNRin=0dB (std) | SNRin=10dB (mean)     |   SNRin=10dB (std) | SNRin=20dB (mean)      |   SNRin=20dB (std) | SNRin=30dB (mean)     |   SNRin=30dB (std) |
|---:|:-----------------------|:----------------------|------------------:|:----------------------|-------------------:|:-----------------------|-------------------:|:----------------------|-------------------:|
|  0 | contour_filtering      | 2.19024311755171      |          0.2524   | 12.339071062253133    |           0.20684  | 23.156231508942156     |           0.245407 | 31.819476738983532    |           0.272845 |
|  1 | delaunay_triangulation | 2.5148629745996036    |          0.757284 | 16.704587492671703    |           0.953092 | 26.183183236161        |           0.798248 | 31.963655849717675    |           5.67399  |
|  2 | empty_space            | 2.8553481960177423    |          0.787573 | 16.615111969904046    |           0.670129 | 25.977539483048826     |           0.58175  | 35.3575906422864      |           0.579961 |
|  3 | hard_thresholding      | **4.885806132900955** |          1.02341  | **18.68011988168515** |           0.653845 | **27.903088500409282** |           0.734602 | **36.87832787402318** |           0.531363 |
### Signal: ExpChirp  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_ExpChirp.html)    [[Get .csv]](.\denoising\csv_files\results_ExpChirp.csv)
|    | Method + Param         | SNRin=0dB (mean)      |   SNRin=0dB (std) | SNRin=10dB (mean)      |   SNRin=10dB (std) | SNRin=20dB (mean)      |   SNRin=20dB (std) | SNRin=30dB (mean)    |   SNRin=30dB (std) |
|---:|:-----------------------|:----------------------|------------------:|:-----------------------|-------------------:|:-----------------------|-------------------:|:---------------------|-------------------:|
|  0 | contour_filtering      | 2.2103732649451135    |          0.197814 | 12.301410620171698     |           0.183853 | 23.17505158582096      |           0.22705  | 31.9164495395881     |           0.286927 |
|  1 | delaunay_triangulation | 3.7035398323611886    |          1.35119  | 16.878033275877627     |           0.97656  | 26.621128905817013     |           0.706433 | 30.275774530188826   |           7.05196  |
|  2 | empty_space            | 4.001876977735019     |          1.29042  | 16.753411233001724     |           0.637199 | 26.430305960372134     |           0.672176 | 36.134673528941306   |           0.534466 |
|  3 | hard_thresholding      | **6.266508314875825** |          1.19612  | **18.828416166657572** |           0.693633 | **28.142227500643795** |           0.56202  | **37.2810726808897** |           0.484208 |
### Signal: ToneSharpAttack  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_ToneSharpAttack.html)    [[Get .csv]](.\denoising\csv_files\results_ToneSharpAttack.csv)
|    | Method + Param         | SNRin=0dB (mean)      |   SNRin=0dB (std) | SNRin=10dB (mean)     |   SNRin=10dB (std) | SNRin=20dB (mean)      |   SNRin=20dB (std) | SNRin=30dB (mean)      |   SNRin=30dB (std) |
|---:|:-----------------------|:----------------------|------------------:|:----------------------|-------------------:|:-----------------------|-------------------:|:-----------------------|-------------------:|
|  0 | contour_filtering      | 2.1909231862353593    |          0.226499 | 12.316574250162805    |           0.28404  | 23.176282602311286     |           0.823508 | 31.575690057421888     |           1.20496  |
|  1 | delaunay_triangulation | 6.8122611475765895    |          1.45931  | 14.42483650719148     |           0.852435 | 23.00749327022713      |           1.04288  | 31.454371479675505     |           0.658233 |
|  2 | empty_space            | 6.683767964713626     |          1.10927  | 14.397076241428442    |           0.806682 | 23.735036391586554     |           1.08298  | 33.39559908453517      |           0.583782 |
|  3 | hard_thresholding      | **9.634822651147264** |          0.759398 | **16.84539135962457** |           0.579168 | **26.615352008921242** |           0.617364 | **36.678418985867914** |           0.627155 |
### Signal: ToneDumped  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_ToneDumped.html)    [[Get .csv]](.\denoising\csv_files\results_ToneDumped.csv)
|    | Method + Param         | SNRin=0dB (mean)       |   SNRin=0dB (std) | SNRin=10dB (mean)     |   SNRin=10dB (std) | SNRin=20dB (mean)     |   SNRin=20dB (std) | SNRin=30dB (mean)      |   SNRin=30dB (std) |
|---:|:-----------------------|:-----------------------|------------------:|:----------------------|-------------------:|:----------------------|-------------------:|:-----------------------|-------------------:|
|  0 | contour_filtering      | 2.266156983046504      |          0.259938 | 12.405761116307081    |           0.208591 | 23.400666083234043    |           0.241552 | 31.957374023534353     |           0.256688 |
|  1 | delaunay_triangulation | 7.426764056769986      |          1.63903  | 16.89123171654579     |           1.09633  | 25.255728459892993    |           5.15464  | 6.261284083079063      |           3.37028  |
|  2 | empty_space            | 7.154812535565789      |          1.03136  | 16.6690923046483      |           0.923835 | 26.624735006358996    |           0.925914 | 36.45605901293928      |           0.874759 |
|  3 | hard_thresholding      | **10.255330357693243** |          1.09351  | **19.09335089830103** |           0.831277 | **28.22545640688133** |           0.637572 | **37.877963784421205** |           0.633376 |
### Signal: McCrossingChirps  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McCrossingChirps.html)    [[Get .csv]](.\denoising\csv_files\results_McCrossingChirps.csv)
|    | Method + Param         | SNRin=0dB (mean)      |   SNRin=0dB (std) | SNRin=10dB (mean)      |   SNRin=10dB (std) | SNRin=20dB (mean)     |   SNRin=20dB (std) | SNRin=30dB (mean)     |   SNRin=30dB (std) |
|---:|:-----------------------|:----------------------|------------------:|:-----------------------|-------------------:|:----------------------|-------------------:|:----------------------|-------------------:|
|  0 | contour_filtering      | **2.181789476954363** |          0.274746 | 12.216849462659738     |           0.194364 | 22.78921515687247     |           0.202091 | 30.90908186825188     |          0.330529  |
|  1 | delaunay_triangulation | 1.4881135744991505    |          0.617819 | 7.778944716158617      |           1.3488   | 10.199886538792853    |           0.772698 | 10.532818686329005    |          0.0536241 |
|  2 | empty_space            | 1.7693496361808614    |          0.707629 | 8.748959472791224      |           1.20314  | 10.599507235965564    |           0.108033 | 10.76891144974018     |          0.0117608 |
|  3 | hard_thresholding      | 1.5305273282430047    |          0.386528 | **15.520982729448622** |           0.575729 | **25.69800744822492** |           0.424176 | **34.92419033671913** |          0.419714  |
### Signal: McMultiLinear  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McMultiLinear.html)    [[Get .csv]](.\denoising\csv_files\results_McMultiLinear.csv)
|    | Method + Param         | SNRin=0dB (mean)       |   SNRin=0dB (std) | SNRin=10dB (mean)      |   SNRin=10dB (std) | SNRin=20dB (mean)      |   SNRin=20dB (std) | SNRin=30dB (mean)     |   SNRin=30dB (std) |
|---:|:-----------------------|:-----------------------|------------------:|:-----------------------|-------------------:|:-----------------------|-------------------:|:----------------------|-------------------:|
|  0 | contour_filtering      | **1.9470244239941403** |         0.167461  | 11.488475711753265     |           0.14689  | 21.36733949171491      |           0.184959 | 28.90340548051425     |           0.269683 |
|  1 | delaunay_triangulation | 1.09670253583037       |         0.37384   | 10.554907341671605     |           1.36968  | 23.037539792770964     |           1.89622  | 32.542254123936914    |           0.225487 |
|  2 | empty_space            | 1.148610068931129      |         0.395619  | **12.078499011795397** |           0.783704 | **23.343271554224067** |           0.293575 | **32.74002549345111** |           0.250211 |
|  3 | hard_thresholding      | 0.09123972719408714    |         0.0828301 | 7.2872050766037475     |           0.562149 | 17.817790429871753     |           0.374289 | 22.106862974174575    |           0.146176 |
### Signal: McCosPlusTone  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McCosPlusTone.html)    [[Get .csv]](.\denoising\csv_files\results_McCosPlusTone.csv)
|    | Method + Param         | SNRin=0dB (mean)      |   SNRin=0dB (std) | SNRin=10dB (mean)     |   SNRin=10dB (std) | SNRin=20dB (mean)      |   SNRin=20dB (std) | SNRin=30dB (mean)     |   SNRin=30dB (std) |
|---:|:-----------------------|:----------------------|------------------:|:----------------------|-------------------:|:-----------------------|-------------------:|:----------------------|-------------------:|
|  0 | contour_filtering      | **2.096296241998868** |          0.155362 | 11.920267053176609    |           0.155682 | 21.86264745135377      |           0.164414 | 29.52470452441685     |           0.32706  |
|  1 | delaunay_triangulation | 1.2636413609090003    |          0.422744 | 11.889052278255479    |           1.71805  | **24.101850752713048** |           0.320963 | 31.589673849251437    |           2.49827  |
|  2 | empty_space            | 1.3092134703205067    |          0.482095 | **13.34622829036295** |           1.09866  | 24.07915183665923      |           0.317415 | **33.40026194342088** |           0.347112 |
|  3 | hard_thresholding      | 0.2545988648535751    |          0.149827 | 11.624891560735158    |           0.546954 | 23.02207055518867      |           0.31186  | 32.48748614079388     |           0.354431 |
### Signal: McMultiCos2  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McMultiCos2.html)    [[Get .csv]](.\denoising\csv_files\results_McMultiCos2.csv)
|    | Method + Param         | SNRin=0dB (mean)      |   SNRin=0dB (std) | SNRin=10dB (mean)      |   SNRin=10dB (std) | SNRin=20dB (mean)     |   SNRin=20dB (std) | SNRin=30dB (mean)     |   SNRin=30dB (std) |
|---:|:-----------------------|:----------------------|------------------:|:-----------------------|-------------------:|:----------------------|-------------------:|:----------------------|-------------------:|
|  0 | contour_filtering      | **2.119592380630293** |          0.254498 | **11.961850652520333** |           0.211527 | **21.81739354582102** |           0.223462 | 28.88652170773657     |           0.303729 |
|  1 | delaunay_triangulation | 0.8996339549524193    |          0.377844 | 5.8481151297108935     |           0.978    | 11.45992434604771     |           1.31458  | 13.076680028051888    |           1.76318  |
|  2 | empty_space            | 0.9225225825156138    |          0.384292 | 8.376010778017106      |           1.0229   | 16.76306409771073     |           0.668108 | 20.008275593669634    |           0.195928 |
|  3 | hard_thresholding      | 0.2605045083810658    |          0.185351 | 9.72794777749874       |           0.649537 | 21.414797763794425    |           0.398327 | **28.99901444402453** |           0.261845 |
### Signal: McSyntheticMixture2  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McSyntheticMixture2.html)    [[Get .csv]](.\denoising\csv_files\results_McSyntheticMixture2.csv)
|    | Method + Param         | SNRin=0dB (mean)       |   SNRin=0dB (std) | SNRin=10dB (mean)      |   SNRin=10dB (std) | SNRin=20dB (mean)     |   SNRin=20dB (std) | SNRin=30dB (mean)     |   SNRin=30dB (std) |
|---:|:-----------------------|:-----------------------|------------------:|:-----------------------|-------------------:|:----------------------|-------------------:|:----------------------|-------------------:|
|  0 | contour_filtering      | **2.1877304930275527** |          0.258623 | 12.324059224767971     |           0.180087 | 22.736276603958846    |           0.262196 | 30.343629547401427    |          0.362808  |
|  1 | delaunay_triangulation | 0.5379083911139615     |          0.311554 | 3.4247265225855235     |           0.515592 | 5.127384925381047     |           0.413437 | 5.241266469750389     |          0.342818  |
|  2 | empty_space            | 0.4953930842716002     |          0.365862 | 4.4420108538485366     |           0.568102 | 6.2048423971386715    |           0.147105 | 6.430077471920219     |          0.0648372 |
|  3 | hard_thresholding      | 0.36186655262912704    |          0.190375 | **12.439744444679196** |           0.510207 | **23.46704005765447** |           0.353051 | **32.40492279076211** |          0.344349  |
### Signal: HermiteFunction  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_HermiteFunction.html)    [[Get .csv]](.\denoising\csv_files\results_HermiteFunction.csv)
|    | Method + Param         | SNRin=0dB (mean)      |   SNRin=0dB (std) | SNRin=10dB (mean)      |   SNRin=10dB (std) | SNRin=20dB (mean)      |   SNRin=20dB (std) | SNRin=30dB (mean)    |   SNRin=30dB (std) |
|---:|:-----------------------|:----------------------|------------------:|:-----------------------|-------------------:|:-----------------------|-------------------:|:---------------------|-------------------:|
|  0 | contour_filtering      | 2.27015762087983      |          0.254213 | 12.327401096833203     |           0.205772 | 23.049534580374424     |           0.202082 | 31.684165894039847   |           0.265295 |
|  1 | delaunay_triangulation | 3.166404632806264     |          1.25828  | 17.751809948580977     |           0.996698 | 27.47094180287482      |           0.888298 | 36.943103254946      |           0.739698 |
|  2 | empty_space            | 3.641554675357997     |          1.46616  | 17.22825082413266      |           0.806352 | 26.955278127001165     |           0.934896 | 36.39794136030626    |           0.636146 |
|  3 | hard_thresholding      | **8.410138806799882** |          1.45319  | **20.306693638327996** |           0.789369 | **29.485500993539528** |           0.841147 | **38.6589020720225** |           0.670954 |
### Signal: McTripleImpulse  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McTripleImpulse.html)    [[Get .csv]](.\denoising\csv_files\results_McTripleImpulse.csv)
|    | Method + Param         | SNRin=0dB (mean)      |   SNRin=0dB (std) | SNRin=10dB (mean)     |   SNRin=10dB (std) | SNRin=20dB (mean)      |   SNRin=20dB (std) | SNRin=30dB (mean)     |   SNRin=30dB (std) |
|---:|:-----------------------|:----------------------|------------------:|:----------------------|-------------------:|:-----------------------|-------------------:|:----------------------|-------------------:|
|  0 | contour_filtering      | 2.205674313751332     |          0.349298 | 10.444857842349094    |           0.76975  | 12.084977274030273     |           1.39784  | 11.032505521953174    |           1.18127  |
|  1 | delaunay_triangulation | 2.063833395944314     |          0.576965 | 13.013938159205274    |           1.25654  | 23.819359919449553     |           0.654943 | 33.61282687293212     |           0.383849 |
|  2 | empty_space            | 2.229480602091429     |          0.705826 | 13.593801844341613    |           0.715925 | 24.168955748451708     |           0.473401 | 33.913437431626484    |           0.362491 |
|  3 | hard_thresholding      | **5.442544592377986** |          0.838539 | **16.64158777484204** |           0.624696 | **26.294590663877685** |           0.515767 | **35.92586739166139** |           0.438907 |
### Signal: McOnOffTones  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McOnOffTones.html)    [[Get .csv]](.\denoising\csv_files\results_McOnOffTones.csv)
|    | Method + Param         | SNRin=0dB (mean)       |   SNRin=0dB (std) | SNRin=10dB (mean)      |   SNRin=10dB (std) | SNRin=20dB (mean)     |   SNRin=20dB (std) | SNRin=30dB (mean)     |   SNRin=30dB (std) |
|---:|:-----------------------|:-----------------------|------------------:|:-----------------------|-------------------:|:----------------------|-------------------:|:----------------------|-------------------:|
|  0 | contour_filtering      | 2.178817945530389      |          0.196763 | 12.133427063524307     |           0.226738 | 22.262562172709917    |           0.218647 | 29.532009177933972    |           0.319716 |
|  1 | delaunay_triangulation | 2.0627399144174183     |          0.750042 | 14.503032804397286     |           2.42973  | 24.434449277047552    |           0.412449 | 31.05555542862299     |           4.80153  |
|  2 | empty_space            | **2.2362441492347385** |          0.886553 | **15.494163610728531** |           0.608907 | 24.61311535814302     |           0.367915 | 33.55491030429578     |           0.322655 |
|  3 | hard_thresholding      | 0.969269409330067      |          0.377888 | 15.080210004313786     |           0.571964 | **25.06959285594782** |           0.465125 | **34.08838305887358** |           0.345108 |
### Signal: McOnOff2  [[View Plot]](https://jmiramont.github.io/benchmark-test/results/denoising/figures/html/plot_McOnOff2.html)    [[Get .csv]](.\denoising\csv_files\results_McOnOff2.csv)
|    | Method + Param         | SNRin=0dB (mean)       |   SNRin=0dB (std) | SNRin=10dB (mean)     |   SNRin=10dB (std) | SNRin=20dB (mean)     |   SNRin=20dB (std) | SNRin=30dB (mean)     |   SNRin=30dB (std) |
|---:|:-----------------------|:-----------------------|------------------:|:----------------------|-------------------:|:----------------------|-------------------:|:----------------------|-------------------:|
|  0 | contour_filtering      | **2.1204078941227635** |          0.176303 | 11.991489345933578    |           0.182418 | 22.169058360252293    |           0.25947  | 30.27416605459649     |           0.290857 |
|  1 | delaunay_triangulation | 1.9058671509392144     |          0.740903 | 14.741497248626185    |           1.49057  | 23.56306256516437     |           3.24074  | 31.30407830611661     |           4.109    |
|  2 | empty_space            | 2.0895400099173265     |          0.791938 | **15.27687397548201** |           0.689622 | 24.463913161162804    |           0.430964 | 33.61977159700306     |           0.346141 |
|  3 | hard_thresholding      | 0.8154862029751955     |          0.326004 | 14.550301282241085    |           0.555708 | **24.80772284583023** |           0.452955 | **34.09608631054921** |           0.366519 |
