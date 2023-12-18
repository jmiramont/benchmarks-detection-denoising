clear all
% close all


folder = './';
addpath('./signals_mat');
%% required paths 
addpath(folder);
addpath(strcat([folder 'Brevdo']));
addpath(strcat([folder 'tools']));
addpath(strcat([folder 'synchrosqueezedSTFT']));
addpath(strcat([folder 'PseudoBay']));
addpath(strcat([folder 'mfiles']));

%% Import signal from file (from the SignalBank in python).
% load McDampedCos.mat
% load McCrossingChirps.mat
load McSyntheticMixture5.mat
N = length(x); % The signal has 1024 samples.
x = x.';
Ncomp = double(Ncomp);

% This vector tells the number of components per time sample.
vec_nc = double(vec_nc); 

% Contaminate the signal with real white Gaussian noise.
noise = randn(N,1);
SNRin = 20;
xn = sigmerge(x, noise, SNRin);

%% Apply the method

% ds = 2;
% use_sst = false;
% beta=0.5;
% alpha=0.5;
% div = 4;
% Pnei = 40;
% PneiMask = 40;
% M = [];
% L = 20;
% return_comps = true;
% return_instf = [];
% [X,mask_total] = pb_method(xn, Ncomp, use_sst, ds, beta, alpha, div, Pnei, PneiMask, M, L, return_comps, return_instf);

% Separated components reconstruction
[X,~] = pb_method(xn, Ncomp, [], [], [], [], [], [], [], [], [],true);
% Complete signal reconstruction
[xr,mask_total] = pb_method(xn, Ncomp, [], [], [], [], [], [], [], [], []);


%% Compute the QRF
qrf = 20*log10(norm(x(100:end-100))/norm(x(100:end-100)-xr(100:end-100).'));

%% Compare recovered signal and the original (denoised) one.
figure();
plot(xr,'k','DisplayName','Recovered signal');
hold on; 
plot(x,'--g','DisplayName','Original signal'); 
legend()


%% Apply the method again, but recover separate components.
% [X,mask_total] = pb_method(xn, Ncomp, use_sst, ds, beta, alpha, div, Pnei, PneiMask, M, L,true, return_instf);

[H L] = roundgauss(2*N); 

figure();
for i=1:Ncomp
    subplot(Ncomp+1,2,2*i-1);
    S = tfrsp(comps(i,:).',1:N,2*N,H);
    imagesc((S(1:N+1,:)));
    title('Original Component: '+string(i));
    
    subplot(Ncomp+1,2,2*i);
    S = tfrsp(X(i,:).',1:N,2*N,H);
    imagesc((S(1:N+1,:)));
    title('Recovered Component: '+string(i));
end

subplot(Ncomp+1,2,2*Ncomp+1)
S = tfrsp(sum(comps).',1:N,2*N,H);
imagesc((S(1:N+1,:)));
title('Sum of Original Components: '+string(i));

subplot(Ncomp+1,2,2*Ncomp+2)
S = tfrsp(sum(X).',1:N,2*N,H);
imagesc((S(1:N+1,:)));
title('Sum of Recovered Components: '+string(i));


