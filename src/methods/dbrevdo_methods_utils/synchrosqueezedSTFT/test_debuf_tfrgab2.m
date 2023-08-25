clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Compute the comparative robustness evaluaton for mask estimation and 
%  mode retrieval considering a signal made of 1 linear chirp merged 
%  with a white Gaussian noise
%  
%
%  Authors : D. Fourer (dominique@fourer.fr) and Q. Legros 
%  Date    : 13-feb-2021
%


folder = './';
%% required paths 
addpath(folder);
addpath(strcat([folder 'Brevdo']));
addpath(strcat([folder 'SSA']));
addpath(strcat([folder 'tools']));
addpath(strcat([folder 'synchrosqueezedSTFT']));


%% Load signal (linear chirp)
N     = 500;                        %% signal length
x0    = real(fmlin(N,0.13,0.3));    %% linear chirp
Ncomp = 1;                          %% number of components
 

% TFR - parameters
M       = 500;       %% nombre de bins frequentiels
Mha = round(M/2);
L       = 20;        %% taille de la fenetre d'analyse en samples

SNR = 20;
x = sigmerge(x0, randn(size(x0)), SNR); %% plus precis 


[tfr] = tfrgab(x, M, L);
[tfr2] = tfrgab2(x, M, L);
isequal(tfr,tfr2)

figure;
subplot(1,2,1);
imagesc(abs(tfr));
% colorbar
title('tfrgab')
subplot(1,2,2);
imagesc(abs(tfr2))
% colorbar
title('tfrgab2')

figure(2);
imagesc(abs(tfr-tfr2))
title('l1 error')
colorbar









