function [ s_hat ] = rectfrgab(tfr, L, M, mm)
% [ s_hat ] = recsMW( stfr, w0. T)
%
% signal reconstruction from the Gabor Transform using the simplified
% formula
%
% INPUT:
% tfr      : the synchrosqueezed Gabor Transform
% L        : T * Ts, the length of the Gaussian Window
% M        : number of frequency bins
% mm       : frequencies bins to reconstruct
%
% OUTPUT:
% s_hat    : reconstructed signal
%
% Author: D.Fourer (dominique@fourer.fr)
% Date: 28-08-2015
% Ref: [D. Fourer, J. Harmouche, J. Schmitt, T. Oberlin, S. Meignen, F. Auger and P. Flandrin. The ASTRES Toolbox for Mode Extraction of Non-Stationary Multicomponent Signals. Proc. EUSIPCO 2017, Aug. 2017. Kos Island, Greece.]
% Ref: [D. Fourer and F. Auger. Second-order Time-Reassigned Synchrosqueezing Transform: Application to Draupner Wave Analysis. Proc. EUSIPCO 2019, Coruna, Spain.]
% Ref: [P. Flandrin. Time-frequency/Time-scale analysis.  Wavelet analysis and its applications. Academic Press. vol 10. 1998]

N = size(tfr, 2);

if ~exist('mm', 'var')
 mm = m_axis(M);
end

s_hat = zeros(1, N);
for n = 1:N
 
 s_hat(n) = sum(tfr(:, n) .* exp(1i *2*pi*(n-1)*mm'/M));
end

s_hat = s_hat * sqrt(2*pi) * L/M;

end

