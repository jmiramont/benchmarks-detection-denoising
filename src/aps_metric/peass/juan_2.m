clear all; close all;
%% Import signal from file (from the SignalBank in python).
load cello.mat
x = data(:,1);
N = length(x);
Ncomp = 1;

% Contaminate the signal with real white Gaussian noise.
noise = randn(N,1);
SNRin = 10;
xn = sigmerge(x, noise, SNRin);


APS = APS_wrapper(x,xn,fs);