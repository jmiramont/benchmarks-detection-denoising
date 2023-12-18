clear all; close all;
% addpath('..');
rng(0);
save_figures = false;
% Load the signal
load cello.mat
%fs = 1/7e-6;
xnoise = signal;
N = 2^11;
xnoise = xnoise(length(xnoise)/2-N/2+1:length(xnoise)/2+N/2);

xnoise = xnoise - mean(xnoise);
xnoise = xnoise / max(abs(xnoise));

% Parameters for the STFT.
% SNRin = 40;
% noise = randn(size(xnoise));
% xnoise = sigmerge(xnoise,noise,SNRin);

Nfft = 4096;
fmax = 0.5; % Max. norm. frequency to compute the STFT.
[w,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.
% w = tftb_window(2*T+1,'Kaiser'); 
% w = tftb_window(T+1,'Hanning');
% w = tftb_window(8*2+1,'Gauss');

[F,~,~] = tfrstft(xnoise,1:N,Nfft,w,0);


%% Noise realization:
J = 512;
dT = 3*ceil(T/8);

% Filtering using classification zeros:
% [mask, signal_r, TRI, TRIselected, ceros, F, class, K, Hceros, zeros_hist] =...
%     classified_zeros_denoising(xnoise, 'estimate', J, dT, [0,0],1.5,[],Nfft);
[mask, signal_r, TRI, TRIselected, ceros, F, class, K, Hceros, zeros_hist] =...
    classified_zeros_denoising(xnoise, 'estimate', J, {'gmm','gap'}, [0,0],1.5,Nfft);

% Changes for the figures:
F = flipud(F(1:round(Nfft/2)+1,:));
ceros(:,1) = round(Nfft/2) +1 - ceros(:,1)  ;
zeros_hist = flipud(zeros_hist);
mask = flipud(mask(1:round(Nfft/2)+1,:));
save variables_cello_classif_zeros.mat

%% Show the histogram.
figure()
imagesc(log(zeros_hist)); hold on;
colormap jet
% viscircles(fliplr(ceros),dT*ones(size(ceros,1),1))
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time'); ylabel('frequency')
title('2D histogram of new zeros')

if save_figures
print_figure('../figures/histogram_batsignal.pdf',5,5,'RemoveMargin',true)
end

%% Spectrogram and zeros.
figure()
% subplot(1,2,1)
% imagesc(-log(abs(F))); hold on;
imagesc(abs(F)); hold on;
% plot(ceros(:,2),ceros(:,1),'o','Color','w','MarkerFaceColor','w','MarkerSize',1.5);
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlim([75 N-75])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
title('Spectrogram','Interpreter','latex');
% colormap pink

if save_figures
print_figure('spectrogram_cello.pdf',4.2,5,'RemoveMargin',true)
end

%% Filtering mask
figure()
imagesc(mask)
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time', 'Interpreter','latex'); ylabel('frequency', 'Interpreter','latex')
title('SZC-GMM-GAP','Interpreter','latex');
colormap bone
xlim([75 N-75])
% axis square
if save_figures
print_figure('mask_cello.pdf',4.2,5,'RemoveMargin',true)
end


%% Classified zeros.
figure()
imagesc(-log(abs(F))); hold on;
% triplot(TRIselected,ceros(:,2),ceros(:,1),'c','LineWidth',0.5);
colores = string({'blue';'green';'red'});
symbols = string({'o';'^';'d'});
for i = 1:3
    plot(ceros(class==i,2),...
        ceros(class==i,1),symbols(i),...
        'Color',colores(i),'MarkerFaceColor',colores(i),'MarkerSize',2.5);
end
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time'); ylabel('frequency')
title('Log-spectrogram and zeros')
colormap pink
% axis square



%%
labels = string({'First kind','Second kind','Third kind'});
figure()
for i = 1:3
    X = Hceros(class==i,:);
    C(i,:) = mean(X,1);
    plot(X(:,1),...
         X(:,2),symbols(i),...
        'Color',colores(i),...
        'MarkerFaceColor',colores(i),...
        'MarkerSize',3,...
        'DisplayName',labels(i)); hold on;
end


plot(C(:,1),C(:,2),'o','MarkerFaceColor','c',...
                        'MarkerEdgeColor','m',...
                        'MarkerSize',4,...
                        'DisplayName','Centroids');


grid on;
xlabel('$\Vert B(z,r) \Vert_{1}$ (normalized)','Interpreter','latex');
ylabel('$H_{1}(B(z,r))$ (normalized)','Interpreter','latex');
xticklabels([]); yticklabels([])
xticks([]); yticks([])
legend('boxoff');
legend('Location','southwest');


%%

% clear all;
% load variables_cello_classif_zeros.mat
load mask_DT_155.mat
load mask_DT_165.mat
load mask_DT_175.mat
load mask_DT_185.mat
load mask_DT_195.mat


figure()
imagesc(mask_DT_175);
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time', 'Interpreter','latex'); ylabel('frequency', 'Interpreter','latex');
title('$\ell_{\max}=1.75$', 'Interpreter','latex');
colormap bone
xlim([75 N-75])
% axis square
if save_figures
print_figure('mask_cello_DT_175.pdf',4.2,5,'RemoveMargin',true)
end

figure()
imagesc(mask_DT_185);
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time', 'Interpreter','latex'); ylabel('frequency', 'Interpreter','latex')
title('$\ell_{\max}=1.85$', 'Interpreter','latex');
colormap bone
xlim([75 N-75])
% axis square
if save_figures
print_figure('mask_cello_DT_185.pdf',4.2,5,'RemoveMargin',true)
end

%%
% figure()
% subplot(2,2,1);
% imagesc(mask);
% title('Three types of zeros')
% subplot(2,2,2);
% imagesc(mask_DT_175);
% title('DT - lmax=1.55')
% subplot(2,2,3);
% imagesc(mask_DT_185);
% title('DT - lmax=1.65')
% subplot(2,2,4);
% imagesc(mask_DT_195);
% title(['DT - lmax=1.7' ...
%     '' ...
%     '5'])
% colormap gray
% 
% 
% %%
% [H ~] = roundgauss(2*N); 
% S = tfrsp(signal_r,1:N,2*N,H);
% imagesc(flipud(log(S(1:N+1,:))));
% xticklabels([]); yticklabels([])
% xticks([]); yticks([])
% xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
% title('SST+RD','Interpreter','latex')
% % print_figure('cello_3t.pdf',4,4,'RemoveMargin',true)


