function [ A_hat ] = estim_amplitude( data,tf,LBcgk )
% [ A_hat ] = estim_amplitude( data,tf,LBcgk )
%
% Estimate the amplitude 
%
% 
% INPUT:
% data      : spectrogram
% tf        : ridges position
% LBcgk     : level of noise (expectation)
% maxF      : mode of the normalized model
% 
% OUTPUT:
% A_hat     : estimated amplitude
%
% Author: Q.Legros (quentin.legros@telecom-paris.fr) and D.Fourer
% Date: 1-mar-2021

[Niter,Ncomp] = size(tf);
% Initialization
A_hat = zeros(size(tf));
for Nc = 1:Ncomp
    for t = 1:Niter
        A_hat(t,Nc) = max(data(t,tf(t,Nc)) - LBcgk,0);
    end
end

end

