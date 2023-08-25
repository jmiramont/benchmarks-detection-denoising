function w_hanning = MakeHanning(L)
% 
% Make a Hanning window of size L.
%
% Guoshen Yu
% Version 1, Sept 15, 2006
%
% Remarks:
% - We impose the window size odd so that we can sample at -pi/2, 0 and pi.
if mod(L,2) ~= 1
    error('The window size has to be odd!');
end

Lhalf = (L-1) / 2;

% w_hanning = cos(pi * [-Lhalf : 1 : Lhalf]/(L-1)).^2; 
argumento = (2*pi * [-Lhalf : 1 : Lhalf]/(L-1));
w_hanning = (1+cos(argumento))/2; 
w_hanning = w_hanning';