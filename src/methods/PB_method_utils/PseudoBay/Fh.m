function [ val ] = Fh( m, M, L )
% [ val ] = Fh( omega, T )
%
% Compute values of the Fourier transform of a Gaussian window
%
% 
% INPUT:
% m      : frequency bins to compute
% M      : number of frequency bins to process
% L      : window time spread parameter
%
% OUTPUT:
% val    : Fh(m) values
%
% Author: D.Fourer (dominique.fourer@univ-evry.fr)
% Date: 03-feb-2021


val = exp(-(2*pi*m/M).^2 * L^2);

end

