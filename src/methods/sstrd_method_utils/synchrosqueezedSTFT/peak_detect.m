function [ m ] = peak_detect( Xw, T, w )
% [ m ] = peak_detect( Xw, T, w )
%
%  extract peaks (local maxima) from a TFR 
%
% Author: D.Fourer (dominique@fourer.fr)
% Date: 15-Feb-2019
% Ref: [D. Fourer, J. Harmouche, J. Schmitt, T. Oberlin, S. Meignen, F. Auger and P. Flandrin. The ASTRES Toolbox for Mode Extraction of Non-Stationary Multicomponent Signals. Proc. EUSIPCO 2017, Aug. 2017. Kos Island, Greece.]
% Ref: [D. Fourer and F. Auger. Second-order Time-Reassigned Synchrosqueezing Transform: Application to Draupner Wave Analysis. Proc. EUSIPCO 2019, Coruna, Spain.]

if ~exist('T', 'var')
 T = eps;
end
m = [];
M = length(Xw);

for i = (w+1):(M-w)
  
  if Xw(i) < T
    continue;
  elseif Xw(i) == max(Xw(max(i-w,1):min(i+w,M))) % (Xw(i) > Xw(i+1)) && (Xw(i) > Xw(i-1))
    m = [m i];
  end
end

end

