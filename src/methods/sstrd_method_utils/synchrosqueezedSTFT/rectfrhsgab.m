function [ s_hat ] = rectfrhsgab(stfr, M)
% [ s_hat ] = rectfrhsgab(stfr, L, M)
%
% signal reconstruction from the horizontally synchrosqueezed Gabor Transform
% %
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
% Ref: [D. Fourer and F. Auger. Second-order Time-Reassigned Synchrosqueezing Transform: Application to Draupner Wave Analysis. Proc. EUSIPCO 2019, Coruna, Spain.]

N = size(stfr,2);

if N == M  %% fast reconstruction
  s_hat = ifft(sum(stfr,2)).';
  
else %% fft cannot be used
  warning('Using slow signal reconstruction M != N');
  m = m_axis(M);
  s_hat = zeros(1, N);
  for n = 1:N
    s_hat(n) = 1/M * sum( sum(stfr,2) .* exp(1j * 2*pi * m.' * (n-1)/M));
  end 
end

s_hat = s_hat(:).';
end

