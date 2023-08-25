function [ LBcgk ] = estimB( data, tf, PneiMask)
% [ LBcgk ] = estimB( data,Mask_out )
%
% Estimate the background level outside of the informative regions
%
% 
% INPUT:
% data       : spectrogram
% Mask_out   : binary mask indicating the informative region
%
% OUTPUT:
% LBcgk      : estimated background level (expectation)
%
% Author: Q.Legros (quentin.legros@telecom-paris.fr) and D.Fourer
% Date: 1-mar-2021

[N,M] = size(data);
% figure(2)
% subplot(2,1,1)
% imagesc(data)
data = data(:);

[Mask_out] = compMask(tf,PneiMask,M,1);
Mask_out = Mask_out(1:N,:); 
Mask_out=min(Mask_out,1);
% subplot(2,1,2)
% imagesc(Mask_out)
Mask_out = Mask_out(:);


LBcgk = mean(data(Mask_out==0));




end
