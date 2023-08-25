This repository contains versions 1.0, 1.1, 2.0, and 2.0.1 of PEASS previously available at http://bass-db.gforge.inria.fr/peass/.


# Purpose
PEASS is a Matlab toolbox that computes perceptually motivated objective measures for the evaluation of audio source separation.

Similarly to [BSS Eval](https://gitlab.inria.fr/bass-db/bss_eval), the distortion signal is decomposed into three components: target distortion, interference, artifacts. These components are then used to compute four quality scores, namely OPS (Overall Perceptual Score), TPS (Target-related Perceptual Score), IPS (Interference-related Perceptual Score), APS (Artifact-related Perceptual Score). These scores better correlate with human assessments than the SDR/ISR/SIR/SAR measures of BSS Eval.


# Versions
[Version 2.0.1](v2.0.1) (December 18, 2017) [1,2]  
This version is identical to Version 2.0 except for the replacement of deprecated Matlab functions.

[Version 2.0](v2.0) (November 8, 2011) [1,2]  
This version was used for the [2011 Signal Separation Evaluation Campaign (SiSEC)](http://sisec2011.wiki.irisa.fr/tiki-index91fe.html?page=Audio+source+separation). It includes some changes in the parameters of the decomposition and of PEMO-Q as well as a more consistent training and feature selection procedure, which greatly improve correlation with human assessments and computational speed compared to the two previous versions.

[Version 1.1](v1.1) (September 16, 2011) [1]  
This version replaces PEMO-Q by the underlying [haircell model](http://medi.uni-oldenburg.de/download/demo/adaption-loops/adapt_loop.zip) (free for academic use). Several advanced features of PEMO-Q are not implemented. It provides similar correlation with human assessments as version 1.0 on average, but individual scores (on a scale between 0 and 100) may differ by as much as &plusmn;20.

[Version 1.0](v1.0) (May 25, 2010) [1]  
This version was used for the [2010 Signal Separation Evaluation Campaign (SiSEC)](http://sisec2010.wiki.irisa.fr/tiki-index91fe.html?page=Audio+source+separation). It requires either the demo version or the full version of [PEMO-Q](https://www.hoertech.de/en/326-hoertech/englisch-ht/f-e-products-ht/pemo-q-ht-en/343-pemo-q.html), which is not sold anymore.


# Usage
See usage instructions for [version 2.0.1](v2.0.1), [version 2.0](v2.0), [version 1.1](v1.1), or [version 1.0](v1.0).


# Subjective data, GUI, and examples
This software comes with
- the [PEASS Subjective Database](data): the set of audio samples and the corresponding subjective ratings used to train the software (see [contents](data/readme.txt))
- the [PEASS Listening Test GUI](GUI): a Matlab GUI to perform MUSHRA tests for subjective evaluation of audio source separation following the proposed protocol (see [usage instructions](GUI/README.txt))
- [PEASS Examples](examples/PEASS-Examples.html): a set of audio examples that illustrate the proposed method for decomposing the distortion into specific components.


# References
[1] Valentin Emiya, Emmanuel Vincent, Niklas Harlander, and Volker Hohmann, ["Subjective and objective quality assessment of audio source separation"](https://hal.inria.fr/inria-00567152/document), *IEEE Transactions on Audio, Speech and Language Processing*, 19(7):2046-2057, 2011.  
[2] Emmanuel Vincent, ["Improved perceptual metrics for the evaluation of audio source separation"](https://hal.inria.fr/hal-00653196/document), in *10th Int. Conf. on Latent Variable Analysis and Signal Separation (LVA/ICA)*, 2012.


# License
This software was authored by Valentin Emiya and Emmanuel Vincent and is distributed under the terms of the [GNU Public License version 3](http://www.gnu.org/licenses/gpl.txt), except the files in the directory named "gammatone" which are under Copyright (C) 2002 2003 2006 2007 [AG Medizinische Physik, Universitaet Oldenburg, Germany](http://www.physik.uni-oldenburg.de/docs/medi) (see [gammatone/README.txt](v2.0.1/gammatone/README.txt)).

See also the licenses for the [PEASS Subjective Database](data/license.txt) and the [PEASS Listening Test GUI](GUI/LICENSE.txt).