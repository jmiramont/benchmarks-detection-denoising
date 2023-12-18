clear all
close all

%% Import signal from file
load McSyntheticMixture.mat

N = length(x); % The signal has 1024 samples.
x = x.';
Ncomp = double(Ncomp);

% This vector tells the number of components per time sample 
% (for SSA, not used in this case).
vec_nc = double(vec_nc); 

% Contaminate the signal with real white Gaussian noise.
rng(0);
noise = randn(N,1);
SNRin = 15;
xn = sigmerge(x, noise, SNRin);

% STFT and Spectrogram
[w,T] = roundgauss(2*N,1e-6); % Round Gaussian window.
[F,~,~] = tfrstft(x,1:N,2*N,w,0);
F = F(1:N+1,:);
F = flipud(F);
S = abs(F).^2;
figure(); imagesc(S);
xticks([]); yticks([]);
% print_figure('spec_example.pdf',4,4,'RemoveMargin',true)

%% Filtering using Contours:
[signal_r,mask_contours] = dt_method(xn, 1.2);

% [signal_r, mask] = empty_space_method(xn,0.6,0.6);

T = sqrt(N);
%%
[H L] = roundgauss(2*N); 
figure();
subplot(1,2,1)
S = tfrsp(x,1:N,2*N,H);
imagesc(flipud(S(1:N+1,:)));
subplot(1,2,2)
S = tfrsp(signal_r.',1:N,2*N,H);
imagesc(flipud(S(1:N+1,:)));

%%
Tind = 58;
QRF = 20*log10(norm(x(Tind:end-Tind))/...
    norm(x(Tind:end-Tind)-signal_r(Tind:end-Tind).'));


S = tfrsp(signal_r.',1:N,2*N,H);
figure()
imagesc(flipud(S(1:N+1,:)));
% imagesc(mask_contours);
xticks([1,N/2,N]); yticks([]);
xticklabels([0,0.5,1.0]); yticklabels([]);
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
title('DT - $\ell_{\max}=1.2$','Interpreter','latex')
% print_figure('dt_example_12.pdf',4,4.2,'RemoveMargin',true)



%% Filtering using Contours:
[signal_r,mask_contours] = dt_method(xn, 1.5);


%%
[H L] = roundgauss(2*N); 
figure();
subplot(1,2,1)
S = tfrsp(x,1:N,2*N,H);
imagesc(flipud(S(1:N+1,:)));
subplot(1,2,2)
S = tfrsp(signal_r.',1:N,2*N,H);
imagesc(flipud(S(1:N+1,:)));

%%
Tind = 58;
QRF = 20*log10(norm(x(Tind:end-Tind))/...
    norm(x(Tind:end-Tind)-signal_r(Tind:end-Tind).'));


S = tfrsp(signal_r.',1:N,2*N,H);
figure()
imagesc(flipud(S(1:N+1,:))); hold on;
plot([100,300],[850 850],'r');
plot([100,300],[600 600],'r');
plot([100,100],[600 850],'r');
plot([300,300],[600 850],'r');

% imagesc(mask_contours);
xticks([1,N/2,N]); yticks([]);
xticklabels([0,0.5,1.0]); yticklabels([]);
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
title('DT - $\ell_{\max}=1.5$','Interpreter','latex')
% print_figure('dt_example_15.pdf',4,4.2,'RemoveMargin',true)

%%
normalizar = @(a) (a-min(a))/(max(a)-min(a))-mean((a-min(a))/(max(a)-min(a)));
figure()
plot(3*normalizar(x'),'LineWidth',0.4); hold on;
plot(3*normalizar(signal_r)-3.5,'LineWidth',0.4)
plot(3*normalizar(x')-3*normalizar(signal_r)-7,'Color',[0.3660 0.6740 0.5680],'LineWidth',0.4); hold on;

plot([100,300],[-6 -6],'r');
plot([100,300],[-8 -8],'r');
plot([100,100],[-8 -6],'r');
plot([300,300],[-8 -6],'r');

xlim([1 N])
ylim([-8.5,4])
xticks([10,N/2,N-10]); yticks([-8,-6]); %, -2.5, -1, 1]);
xticklabels([0,0.5,1.0]); yticklabels(string([-1,1]));%,-1,1,-1,1]));
xlabel('time', 'Interpreter','latex');
ylabel('amplitude', 'Interpreter','latex');

legend('$x$','$\tilde{x}$ (DT-$\ell_{\max}=1.5$)','$x-\tilde{x}$','Location','north', 'NumColumns',3,'Box',false,'FontSize',8.0,'Interpreter','latex')
leg = legend();
leg.ItemTokenSize = [10,30];
% if save_figures
% print_figure('figures/mixture_signal_example_dt_ofsset.pdf',8.3,4.0,'RemoveMargin',true)

%%
figure()
plot(x,'--g','LineWidth',0.3); hold on;
plot(signal_r,'k','LineWidth',0.001)
xlim([1 N])
ylim([-4,5.2])
xticks([]); %yticks([]);
% xticklabels([0,0.5,1.0]); %yticklabels([]);
% plot([100,300],[3 3],'r');
% plot([100,300],[-3.5 -3.5],'r');
% plot([100,100],[-3.5 3],'r');
% plot([300,300],[-3.5 3],'r');
% xlabel('time', 'Interpreter','latex');
ylabel('amplitude', 'Interpreter','latex');

legend('Original','Recovered - DT-$\ell_{\max}=1.5$','Location','north', 'NumColumns',2,'Box',false,'FontSize',6.0,'Interpreter','latex')
% if save_figures
% print_figure('figures/mixture_signal_example_dt.pdf',8.3,2.75,'RemoveMargin',true)
% end

%%



figure()
% plot(signal_r,'g','LineWidth',0.25); hold on;
plot(x'-signal_r,'k','LineWidth',0.15); hold on;
xlim([1 N])
ylim([-2.3,2.3])
xticks([1,N/2,N]); %yticks([]);
xticklabels([0,0.5,1.0]); %yticklabels([]);
plot([100,300],[2 2],'r');
plot([100,300],[-2 -2],'r');
plot([100,100],[-2 2],'r');
plot([300,300],[-2 2],'r');
xlabel('time', 'Interpreter','latex');
ylabel('amplitude', 'Interpreter','latex');

% legend('Original','Recovered - DT-$\ell_{\max}=1.5$','Location','north', 'NumColumns',2,'Box',false,'FontSize',6.0,'Interpreter','latex')
% if save_figures
% print_figure('figures/mixture_signal_example_dt_error.pdf',8.3,2.4,'RemoveMargin',true)
% end