PEASS subjective database
Version 1.0, May 12th, 2010.
By Valentin Emiya, INRIA, 2010
This is a joint work by Valentin Emiya (INRIA, France), Emmanuel Vincent (INRIA, France), Niklas Harlander (University of Oldenburg, Germany), Volker Hohmann (University of Oldenburg, Germany).

********
Contents
********

The PEASS subjective database provides a set of sounds and the subjective scores resulting from listening tests.

Ten mixtures (exp01 to exp10) are proposed. For each of them (say expXX), the following audio files are available:
- expXX_target.wav: the target source (also used as a hidden reference);
- expXX_InterfSrc1.wav, expXX_InterfSrc2.wav, ...: the other sources;
- expXX_anchorDistTarget: the anchor sound related to the distortion of the target source;
- expXX_anchorInterf: the anchor sound related to the presence of other sources;
- expXX_anchorArtif: the anchor sound related to the presence of artifact;
- expXX_test5, expXX_test6, expXX_test7, expXX_test8: the estimates of the target source obtained from the mixture by actual source separation algorithms.

Note that:
- the evaluated mixture is the sum of the target source and the other sources;
- from one mixture to another, the estimates of the target source are not obtained by the same source separation algorithms.

The subjective scores are available in the MATLAB file called PEASS-subjdata.mat. It includes the following variables:
- scores: the 320 scores obtained for each of the 20 reliable subjects;
- IRef: a 320-length boolean vector in which true values indicates the score indices for the grading of hidden references;
- IAnchorTarget: a 320-length boolean vector in which true values indicates the score indices for the grading of anchor sounds related to the distortion of the target source;
- IAnchorInterf: a 320-length boolean vector in which true values indicates the score indices for the grading of anchor sounds related to the presence of other sources;
- IAnchorArtif: a 320-length boolean vector in which true values indicates the score indices for the grading of anchor sounds related to the presence of artifact;
- IGlobalScore: a 320-length boolean vector in which true values indicates the score indices for test 1 about the global quality;
- ITargetPreservationScore: a 320-length boolean vector in which true values indicates the score indices for test 2 about the preservation of the target source;
- IOtherSourceSuppressionScore: a 320-length boolean vector in which true values indicates the score indices for test 3 about the presence of other sources;
- IArtificialNoiseAbsenceScore: a 320-length boolean vector in which true values indicates the score indices for test 4 about the absence of additional artificial noise;
- NMix (=10): the number of mixtures/trials;
- NSubjects (=20): the number of reliable subjects;
- NTasks (=4): the number of tasks/tests;
- NScoredSounds (=8): the number of sounds scored for each mixture (1 hidden reference, 3 anchors, 4 estimates by source separation algorithms);
- outlierScores: the 320 scores obtained for each of the 3 outlier subjects (unreliable subjects);
- soundNames: a 320-length cell array with the sound names related to the 320 scores.

******************************************
How to cite the PEASS Subjective Database?
******************************************
When using this database, the following paper must be referred to:

Valentin Emiya, Emmanuel Vincent, Niklas Harlander and Volker Hohmann, Subjective and objective quality assessment of audio source separation, IEEE Transactions on Audio, Speech and Language Processing, submitted.

*********
Copyright
*********
The files in root directory are under:
Copyright 2010 Valentin Emiya (INRIA).

Licenses: see license.txt.
