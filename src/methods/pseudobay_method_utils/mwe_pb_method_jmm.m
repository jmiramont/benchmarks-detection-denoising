%% Minimum Working Example PB Method
% A two-component signal is created and the method is applied, obtaining a
% number of masks, one per component, the sum of all produces a total mask
% for denoising the stft.

clear all
close all

folder = './';
%% required paths 
addpath(folder);
% addpath(strcat([folder 'Brevdo']));
addpath(strcat([folder 'tools']));
addpath(strcat([folder 'synchrosqueezedSTFT']));
addpath(strcat([folder 'PseudoBay']));

%% 1: Create signal
N = 2^10;
time=(1:N)';
s=zeros(N,2);
% s1
lambda0=0.03;
s(:,1) = cos(2*pi*lambda0*time);
s(:,1) = s(:,1) - mean(s(:,1));
% s2
lambda1     = 0.06;
deltalambda = (0.15-0.06);
s(:,2)  = 0.8 * cos(2*pi * (lambda1 * (time) + (deltalambda/(2*N)) * (time).^2));
s(:,2) = s(:,2) - mean(s(:,2));
% mix: x= s1 + s2
x = s(:,1)+s(:,2);
% noise with arbitrary std value
noise = randn(N,1);
% mix + noise
SNRin = 5;
xn = sigmerge(x,noise,SNRin);
Ncomp =2;


% Apply method.
xr = pb_method(xn,Ncomp,true,[],[],[],[],30).';

QRF = 20*log10(norm(x(1:end))/norm(x(1:end)-xr(1:end)));

figure();
plot(x,'g--'); hold on; plot(xr,'k');