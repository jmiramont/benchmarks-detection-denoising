% Example 2 - bat signal analysis
clear all
close all

folder = './';
%% required paths 
addpath(folder);
addpath(strcat([folder 'synchrosqueezedSTFT']));
addpath(strcat([folder 'tools']));
addpath(strcat([folder 'PseudoBay']));

Ncomp = 3;

load('bat2.mat');

N = length(x);
L = 9;

M = 512;
M2 = floor(M/2);

t = ((1:N)-1)/Fs*1000; %converted in ms
f = m_axis(M)/M*Fs;


[tfr,stfr]  = tfrsgab2(x, M, L);
spect=abs(tfr(1:M2,:)).^2;
figure(1)
plot_tfr(spect,t,f(1:M2))
xlabel('Time [ms]','FontSize', 12, 'FontWeight', 'bold')
ylabel('Frequency [Hz]','FontSize', 12, 'FontWeight', 'bold')

%% Bayesian method parameters
ds    = 3;    % variance of the random walk in the temporal model

beta  = 0.7;
alpha = 0.3;
div   = 4;                         % 1 = KL
                                   % 2 = beta
                                   % 3 = Renyi
                                   % 4 = AB-div
Pnei = 8;
PneiMask = 4;
detect = 0;   
ifplot =  0;

[mask,~] = pseudoBay(tfr,Ncomp, M, L, div, beta, alpha, ds, Pnei, ifplot, detect, PneiMask);

cols = {'r-', 'g-', 'b-', 'k-', 'm-x', 'g-x', 'w-o'};
figure(1)
hold on
for c = 1:Ncomp
  [ IF ] = mask2if( mask(1:M2,:,c) );
  h(c) = plot(t,IF/M*Fs, cols{c});
  label{c} = sprintf('mode %d', c);
end
legend(h, label);
axis square




% With detect
detect = 1;   
ifplot =  0;
[mask,~] = pseudoBay(tfr,Ncomp, M, L, div, beta, alpha, ds, Pnei, ifplot, detect, PneiMask);

cols = {'r-', 'g-', 'b-', 'k-', 'm-x', 'g-x', 'w-o'};
figure(2)
plot_tfr(spect,t,f(1:M2))
xlabel('Time [ms]','FontSize', 12, 'FontWeight', 'bold')
ylabel('Frequency [Hz]','FontSize', 12, 'FontWeight', 'bold')
hold on
for c = 1:Ncomp
  [ IF ] = mask2if( mask(1:M2,:,c) );
  h(c) = plot(t,IF/M*Fs, cols{c});
  label{c} = sprintf('mode %d', c);
end
legend(h, label);
axis square





