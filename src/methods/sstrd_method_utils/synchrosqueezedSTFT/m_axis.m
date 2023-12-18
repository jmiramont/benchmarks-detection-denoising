function [ mm ] = m_axis(M, i)
% [ mm ] = m_axis(M, i)
% Compute the frequency bins in range {0} U [1, M/2] U [-M/2+1, -1]
% 
% INPUT:
% M      : number of frequency bins to process
% i      : (optional) bin selection
%
% OUTPUT:
% mm     : the computed frequency axis 
%
% Author: D.Fourer
% Date: 28-08-2015
% Ref: [D. Fourer, J. Harmouche, J. Schmitt, T. Oberlin, S. Meignen, F. Auger and P. Flandrin. The ASTRES Toolbox for Mode Extraction of Non-Stationary Multicomponent Signals. Proc. EUSIPCO 2017, Aug. 2017. Kos Island, Greece.]
% Ref: [D. Fourer and F. Auger. Second-order Time-Reassigned Synchrosqueezing Transform: Application to Draupner Wave Analysis. Proc. EUSIPCO 2019, Coruna, Spain.]

%method = 0;   % [0, M/2-1] U [-M/2+1, 0]        %% provide better results when summing both positive and negative frequencies

%%%%%%%%%   mm = (1:M)-1;

method = 1;  %{0} U [1, M/2] U [-M/2+1, -1]


Mh = floor(M/2);
mm = 1:M;

if method == 1
  I1 = find(mm <= (Mh+1));
  I2 = find(mm > (Mh+1));
  mm(I1) = mm(I1)-1;
  mm(I2) = -(M+1)+mm(I2);
else
  I1 = find(mm <= (Mh));
  I2 = find(mm > (Mh));
  mm(I1) = mm(I1)-1;
  mm(I2) = -(M)+mm(I2);
end


if exist('i', 'var')
 mm = mm(i); 
end
%mm = -mm;
end

