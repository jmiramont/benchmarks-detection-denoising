function signal_r = ssa_denoising(signal, L, n_components, epsilon)
%METHOD_SSA_DECOMP Summary of this function goes here
%   Detailed explanation goes here


if nargin < 2
L  = 40;  %% embedded dimension SSA parameter
end

if nargin < 3
n_components = 2;   %% number of components
end

if nargin < 4
epsilon = 5e-3;    %% singular spectrum thresholding parameter (increase for more robustness to noise)
end
disp('inargs')
disp([L,n_components,epsilon])
disp('inargs2')
N  = length(signal);
Y = ssa_decomp(signal, L, n_components, epsilon);

signal_r = sum(Y,2).';
end

