function [ s_hat ] = rectfrsgab(stfr, L, M)
% [ s_hat ] = recsMW( stfr, w0. T)
%
% signal reconstruction from the synchrosqueezed Gabor Transform
% formala
%
% INPUT:
% stfr     : the synchrosqueezed Gabor Transform
% L        : T * Ts, the length of the Gaussian Window
% M        : number of frequency bins
%
% OUTPUT:
% s_hat  : reconstructed signal
%
% Author: D.Fourer (dominique@fourer.fr)
% Date: 28-08-2015
% Ref: [D. Fourer, J. Harmouche, J. Schmitt, T. Oberlin, S. Meignen, F. Auger and P. Flandrin. The ASTRES Toolbox for Mode Extraction of Non-Stationary Multicomponent Signals. Proc. EUSIPCO 2017, Aug. 2017. Kos Island, Greece.]
% Ref: [D. Fourer and F. Auger. Second-order Time-Reassigned Synchrosqueezing Transform: Application to Draupner Wave Analysis. Proc. EUSIPCO 2019, Coruna, Spain.]% Ref: [P. Flandrin. Time-frequency/Time-scale analysis.  Wavelet analysis and its applications. Academic Press. vol 10. 1998]

s_hat = sum(stfr) * (2*pi)^(3/2) * L/M;

end

