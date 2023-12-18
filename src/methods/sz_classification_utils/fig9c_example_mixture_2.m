% Classification of spectrogram zeros. Example with different types of
% modes.

clear all; close all;
save_figures = false; % If true, save the figures to .pdf.
rng(0)

% Generate the signal
N = 2^10;
load McSyntheticMixture.mat
% load McMultiLinear2.mat
x = x.';

% Parameters for the STFT.
Nfft = 2*N;
fmax = 0.5; % Max. norm. frequency to compute the STFT.
[w,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.

% Noise realization:
J = 256;
r = 3*ceil(T/8);
SNRin = 20;
noise = randn(size(x));
xnoise = sigmerge(x,noise,SNRin);

%% Filtering using classification zeros:
tic()
[mask, signal_r, TRI, TRIselected, ceros, F, class, K, Hceros, zeros_hist] =...
    classified_zeros_denoising(xnoise, 'estimate', J, {'gmm','gap'}, [0,0],1.0);
toc()
% Changes for the figures:
F = flipud(F(1:N+1,:));
ceros(:,1) = N +1 - ceros(:,1);
zeros_hist = flipud(zeros_hist);
mask = flipud(mask(1:N+1,:));


%%
normalizar = @(a) (a-min(a))/(max(a)-min(a))-mean((a-min(a))/(max(a)-min(a)));
figure()
plot(3*normalizar(x),'LineWidth',0.4); hold on;
plot(3*normalizar(signal_r)-3.5,'LineWidth',0.4)
plot(3*normalizar(x)-3*normalizar(signal_r)-7,'Color',[0.3660 0.6740 0.5680],'LineWidth',0.4); hold on;
xlim([1 N])
ylim([-8.5,4])
xticks([]); yticks([-8,-6]); %, -2.5, -1, 1]);
xticklabels([0,0.5,1.0]); yticklabels(string([-1,1]));%,-1,1,-1,1]));
% xlabel('time', 'Interpreter','latex');
ylabel('amplitude', 'Interpreter','latex');

% legend('Original','Recovered','Location','north', 'NumColumns',2,'Box',false,'FontSize',6.0,'Interpreter','latex')
legend('$x$','$\tilde{x}$ (SZC-GMM-GAP)','$x-\tilde{x}$','Location','north', 'NumColumns',3,'Box',false,'FontSize',8.0,'Interpreter','latex')
leg = legend();
leg.ItemTokenSize = [10,30];
% if save_figures
% print_figure('figures/mixture_signal_example_offset.pdf',8.3,3.5,'RemoveMargin',true)


%%
figure()
plot(x,'--g','LineWidth',0.1); hold on;
plot(signal_r,'k','LineWidth',0.001)
xlim([1 N])
ylim([-4,5.2])
xticks([]); %yticks([]);
xticklabels([0,0.5,1.0]); %yticklabels([]);
% xlabel('time', 'Interpreter','latex');
ylabel('amplitude', 'Interpreter','latex');

legend('Original','Recovered - SZC-GMM-GAP','Location','north', 'NumColumns',2,'Box',false,'FontSize',6.0,'Interpreter','latex')
% if save_figures
% print_figure('figures/mixture_signal_example.pdf',8.3,2.75,'RemoveMargin',true)
% end

%%
figure()
% plot(signal_r,'g','LineWidth',0.25); hold on;
plot(x-signal_r,'k','LineWidth',0.15); hold on;
xlim([1 N])
ylim([-2.3,2.3])
xticks([]); %yticks([]);
xticklabels([0,0.5,1.0]); %yticklabels([]);
% plot([100,300],[2 2],'r');
% plot([100,300],[-2 -2],'r');
% plot([100,100],[-2 2],'r');
% plot([300,300],[-2 2],'r');
% xlabel('time', 'Interpreter','latex');
ylabel('amplitude', 'Interpreter','latex');

% legend('Original','Recovered - DT-$\ell_{\max}=1.5$','Location','north', 'NumColumns',2,'Box',false,'FontSize',6.0,'Interpreter','latex')
% if save_figures
print_figure('figures/mixture_signal_example_error.pdf',8.3,1.8,'RemoveMargin',true)
% end

%%
% Filtering using classification zeros:
% [mask, signal_r, TRIselected, TRIselected_2, ceros, F, class, K, Hceros, zeros_hist] =...
%     classified_zeros_denoising_2(xnoise, 'estimate', J, r, 1.4, [5,5],[],[],Nfft);
% signal_r_2 = signal_r(:,2);
% signal_r = signal_r(:,1);
% mask2 = mask(:,:,2);
% mask = mask(:,:,1);
% 
% mask2 = flipud(mask2(1:round(Nfft/2)+1,:));
% QRF2 = 20*log10(norm(x)/norm(x-signal_r_2));

%%
% Changes for the figures:
% F = flipud(F(1:round(Nfft/2)+1,:));
% ceros(:,1) = round(Nfft/2) +1 - ceros(:,1)  ;
zeros_hist = flipud(zeros_hist);
mask = flipud(mask(1:round(Nfft/2)+1,:));
QRF = 20*log10(norm(x(2*T:end-2*T))/norm(x(2*T:end-2*T)-signal_r(2*T:end-2*T)));


%% Circles with the patch radius on the histogram:
figure()
imagesc((zeros_hist).^0.3); hold on;
colormap jet
u=ceros(:,1);
v=ceros(:,2);
[vx,vy] = voronoi(u,v);
figure()
imagesc(log(zeros_hist+eps)); hold on;
plot(round(vy),round(vx),'w')
% % plot(vy,vx,'og','MarkerFaceColor','g')
plot(ceros(:,2),ceros(:,1),'o','Color','r','MarkerFaceColor','r','MarkerSize',4); hold on
% viscircles(fliplr(ceros),r*ones(size(ceros,1),1))
% text(ceros(:,1),ceros(:,2), string(1:length(ceros)),'HorizontalAlignment','center');

%% Spectrogram and zeros.
figure()
% subplot(1,2,1)
imagesc(-log(abs(F))); hold on;
plot(ceros(:,2),ceros(:,1),'o','Color','w','MarkerFaceColor','w','MarkerSize',2);
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
title('Spectrogram and zeros','Interpreter','latex')
colormap pink

if save_figures
    print_figure('figures/spectrogram_mixture.pdf',7,4,'RemoveMargin',true)
end

%% Filtering mask
figure()
imagesc(mask)
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
title('Extraction Mask')
colormap bone
% axis square

%% Classified zeros.
figure()
% imagesc(-log(abs(F))); hold on;
imagesc((abs(F))); hold on;
% triplot(TRIselected,ceros(:,2),ceros(:,1),'c','LineWidth',0.5);
colores = string({'c';'green';'red'});
symbols = string({'o';'^';'d'});
for i = 1:3
    plot(ceros(class==i,2),...
        ceros(class==i,1),symbols(i),...
        'Color',colores(i),'MarkerFaceColor',colores(i),'MarkerSize',2);
end
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
% title('Log-Spectrogram and zeros','Interpreter','latex')
% colormap pink
% axis square

% print_figure('example_ascete_mixutre.pdf',5,5,'RemoveMargin',true)

if save_figures
    print_figure('figures/mixture_2.pdf',8.3,4,'RemoveMargin',true)
end

%%
labels = string({'SS','NN','SN'});

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

% text(Hceros(:,1),Hceros(:,2), string(1:length(Hceros)));

plot(C(:,1),C(:,2),'o','MarkerFaceColor','c',...
                        'MarkerEdgeColor','m',...
                        'MarkerSize',4,...
                        'DisplayName','Centroids');


grid on;
xlabel('$\Vert G_{z} \Vert_{1}$','Interpreter','latex');
ylabel('$H_{\infty}(G_{z})$','Interpreter','latex');
xticklabels([]); yticklabels([])
xticks([]); yticks([])
legend('boxoff');
legend('Location','southwest');

if save_figures
print_figure('figures/feature_space_mixutre.pdf',7,5,'RemoveMargin',true)
end


%%
figure()
[H L] = roundgauss(2*N); 
S = tfrsp(signal_r,1:N,2*N,H);
imagesc(flipud(S(1:N+1,:)));
% imagesc(mask_contours);
xticks([1,N/2,N]); yticks([]);
xticklabels([0,0.5,1.0]); yticklabels([]);

xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
title('SZC-GMM-GAP','Interpreter','latex')
% print_figure('3t_example.pdf',4,4.2,'RemoveMargin',true)
