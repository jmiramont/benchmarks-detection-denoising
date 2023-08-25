%	
%	Script test generating the time-frequency representations 
%   of a synthetic signal. 
%   /!\ This raw version does not use FFT and is slower
%
%   The figures correspond to Fig.1 presented in following paper:
%
%  [D.Fourer and F. Auger, Second-order Time-Reassigned Synchrosqueezing
%  Transform: Application to Draupner Wave Analysis, submitted to EUSIPCO 19]
%
%
%	D. Fourer, Fev. 2019
%   Copyright (c) 2019 by CNRS (France). 
% 
%	------------------- CONFIDENTIAL PROGRAM --------------------
% 	This program can not be used without the authorization of its 
% 	author(s). For any comment or bug report, please send e-mail to 
% 	dominique@fourer.fr

clear all
close all

chemin = 'figs';                 %location where the computed figures are stored               

Fs = 1000;                       %sampling frequency
snr_val = 25; %[45 35 25 15 5];  %% snr values to compute

signal = 1;                      % change value in [1,2,3,4]

gamma_K = 1e-5;                  %accuracy of the computed transform
M = 600;
L = 8;
M2 = round(M/2);
alpha = 0.25;

zpad = 50;          % zero padding of the signal

q_method = 3;       %% Local modulation estimator to use: 2 => t2,  3 =>w2

if ~exist(chemin, 'dir')
 mkdir(chemin)    
end

%% 1 - generate signal
load_signal
n  = 1:length(s);   %% time sample
N  = length(n);     %% signal length
s0 = s-mean(s);     %% 0-mean signal
%s0 = s;

s = sigmerge(s0, randn(size(s0)), snr_val);
rsb = SNR(s0, s);

sz       = [zeros(zpad,1);s;zeros(zpad,1)];  %% zero-padded signal
n_range = (zpad+1):(zpad+length(s0));

nfreqs = ((1:M2)-1)/M;

%% 1 - spectrogram  / 2 - reassigned spectrogram
[tfr, rtfrg] = tfrrgab(s, M, L, gamma_K);

figure(1)
imagesc(n, nfreqs, abs(tfr(1:M2,:).^2).^alpha);
ylim([0 0.499])
set(gca,'YDir','normal')
xlabel('time samples', 'FontSize', 16)
ylabel('normalized frequency', 'FontSize', 16)
title(sprintf('SNR=%0.02f dB, L=%0.02f', snr_val, L), 'FontSize', 14);
colormap gray;
cmap = colormap;
cmap = flipud(cmap);
colormap(cmap);
saveas(gcf, sprintf('%s/spectrogram.eps', chemin), 'epsc');


[ s_hat ] = real(rectfrgab(tfr, L, M));

figure(11)
plot(s)
hold on
plot(s_hat, 'r-.')
legend('ref', 'reconstruction')
rqf_s = RQF(s.',s_hat);
title(sprintf('Signal RQF=%.2f dB', rqf_s), 'FontSize', 14);
saveas(gcf, sprintf('%s/reconstruction-stft.eps', chemin), 'epsc');


figure(2)
imagesc(n, nfreqs, rtfrg(1:M2,:).^alpha);
ylim([0 0.499])
set(gca,'YDir','normal')
xlabel('time samples', 'FontSize', 16)
ylabel('normalized frequency', 'FontSize', 16)
title(sprintf('SNR=%0.02f dB, L=%0.02f', snr_val, L), 'FontSize', 14);
colormap gray;
cmap = colormap;
cmap = flipud(cmap);
colormap(cmap);
saveas(gcf, sprintf('%s/reassigned-spectrogram.eps', chemin), 'epsc');



%% 3 - time-reassigned synchrosqueezed STFT
[~, stfr, lost, lost2] = tfrtsgab(sz, M, L, gamma_K);

figure(3)
imagesc(n, nfreqs, abs(stfr(1:M2,n_range).^2).^alpha);
ylim([0 0.499])
set(gca,'YDir','normal')
xlabel('time samples', 'FontSize', 16)
ylabel('normalized frequency', 'FontSize', 16)
title(sprintf('SNR=%0.02f dB, L=%0.02f', snr_val, L), 'FontSize', 14);
colormap gray;
cmap = colormap;
cmap = flipud(cmap);
colormap(cmap);
saveas(gcf, sprintf('%s/time-reassigned-SST.eps', chemin), 'epsc');



stfr(:,end) = stfr(:,end) + lost2;  %% recover lost energy to improve reconstruction
%% reconstruct signal
s_hat = real(rectfrhsgab(stfr, M));
s_hat = s_hat(n_range);

figure(31)
plot(s)
hold on
plot(s_hat, 'r-.')
legend('ref', 'reconstruction')
rqf_s = RQF(s.',s_hat);
title(sprintf('Signal RQF=%.2f dB', rqf_s), 'FontSize', 14);
saveas(gcf, sprintf('%s/reconstruction_time-reassigned-SST.eps', chemin), 'epsc');


%% 4 - second-order horizontal synchrosqueezed STFT
[~, stfr2, lost, lost2] = tfrthsgab(sz, M, L,gamma_K, q_method);

alpha2 = 0.35;
figure %(4)
imagesc(n, nfreqs, abs(stfr2(1:M2,n_range).^2).^alpha2);
ylim([0 0.499])
set(gca,'YDir','normal')
xlabel('time samples', 'FontSize', 16)
ylabel('normalized frequency', 'FontSize', 16)
title(sprintf('SNR=%0.02f dB, L=%0.02f', snr_val, L), 'FontSize', 14);
colormap gray;
cmap = colormap;
cmap = flipud(cmap);
colormap(cmap);
saveas(gcf, sprintf('%s/second-order-horizontal_time-reassigned-SST.eps', chemin), 'epsc');

%% reconstruct signal
stfr2(:,end) = stfr2(:,end) + lost2;
s_hat = real(rectfrhsgab(stfr2, M));
s_hat = s_hat(n_range);

figure(42)
plot(s)
hold on
plot(s_hat, 'r-.')
legend('ref', 'reconstruction')
rqf_s = RQF(s.',s_hat);
title(sprintf('Signal RQF=%.2f dB', rqf_s), 'FontSize', 14);
saveas(gcf, sprintf('%s/reconstruction_second-order-horizontal_time-reassigned-SST.eps', chemin), 'epsc');



%% 5 - (classical) synchrosqueezed STFT
[~, stfr, lost] = tfrsgab(s, M, L, gamma_K);

figure(5)
imagesc(n, nfreqs, abs(stfr(1:M2,:).^2).^alpha);
ylim([0 0.499])
set(gca,'YDir','normal')
xlabel('time samples', 'FontSize', 16)
ylabel('normalized frequency', 'FontSize', 16)
title(sprintf('SNR=%0.02f dB, L=%0.02f', snr_val, L), 'FontSize', 14);
colormap gray;
cmap = colormap;
cmap = flipud(cmap);
colormap(cmap);
saveas(gcf, sprintf('%s/synchrosqueezing.eps', chemin), 'epsc');


%% reconstruct signal
s_hat = real(rectfrsgab(stfr, L, M));
figure(51)
plot(s)
hold on
plot(s_hat, 'r-.')
legend('ref', 'reconstruction')
rqf_s = RQF(s.',s_hat);
title(sprintf('Signal RQF=%.2f dB', rqf_s), 'FontSize', 14);
saveas(gcf, sprintf('%s/reconstruction_SST.eps', chemin), 'epsc');


%% 6 - second-order vertically synchrosqueezed STFT
[tfr, stfr2, lost, ~, ~] = tfrvsgab(s, M, L, q_method, 2, gamma_K, gamma_K);


figure(6)
imagesc(n, nfreqs, abs(stfr2(1:M2,:).^2).^alpha);
ylim([0 0.499])
set(gca,'YDir','normal')
xlabel('time samples', 'FontSize', 16)
ylabel('normalized frequency', 'FontSize', 16)
title(sprintf('SNR=%0.02f dB, L=%0.02f', snr_val, L), 'FontSize', 14);
colormap gray;
cmap = colormap;
cmap = flipud(cmap);
colormap(cmap);
saveas(gcf, sprintf('%s/second-order-vertical-SST.eps', chemin), 'epsc');


%% reconstruct signal
s_hat = real(rectfrsgab(stfr2, L, M));
figure(61)
plot(s)
hold on
plot(s_hat, 'r-.')
legend('ref', 'reconstruction')
rqf_s = RQF(s.',s_hat);
title(sprintf('Signal RQF=%.2f dB', rqf_s), 'FontSize', 14);
saveas(gcf, sprintf('%s/reconstruction_second-order-vertical-SST.eps', chemin), 'epsc');

eps2pdf(chemin);