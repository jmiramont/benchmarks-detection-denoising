% Classification of spectrogram zeros. Example with different types of
% modes.

clear all; %close all;
save_figures = false; % If true, save the figures to .pdf.
rng(0)

% Generate the signal
N = 2^10;
tmargin = round(sqrt(N));
Nchirp = N-4*tmargin;
tchirp = 0:Nchirp-1;
impulse_loc = tmargin;
impulse_amp = 10;

f_init = 0.18;
f_end = 0.20;
lin_chirp_instf = f_init + (f_end-f_init)*tchirp/Nchirp;
lin_chirp = cos(2*pi*cumsum(lin_chirp_instf));
cos_chirp_instf = 0.3 + 0.08*cos(2*pi*2*tchirp/Nchirp);
cos_chirp = cos(2*pi*cumsum(cos_chirp_instf));
chirp_part = 1.25*cos_chirp + lin_chirp;
impulse = zeros(1,2*tmargin); impulse(impulse_loc) = impulse_amp;
x = [zeros(1,2*tmargin) chirp_part.*tukeywin(length(chirp_part),0.25).' zeros(1,2*tmargin)];
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
[xnoise,h] = sigmerge(x,noise,SNRin);

% Filtering using classification zeros:
[mask, signal_r, TRI, TRIselected, zeros_pos, F, class, K, Hceros, zeros_hist] =...
    classified_zeros_denoising(xnoise, 'estimate', J, {'gmm','gap'}, [0,0]);


% Changes for the figures:
F = flipud(F(1:N+1,:));
zeros_pos(:,1) = N +1 - zeros_pos(:,1)  ;
zeros_hist = flipud(zeros_hist);
mask = flipud(mask(1:N+1,:));

QRF = 20*log10(norm(x)/norm(x-signal_r));


%% Circles with the patch radius on the histogram:
figure()
imagesc(zeros_hist.^0.3, "AlphaData",1); hold on;
colormap jet
plot(zeros_pos(:,2),zeros_pos(:,1),'o','Color','r','MarkerFaceColor','r',...
    'MarkerSize',4); hold on;
% viscircles(fliplr(zeros_pos),r*ones(size(zeros_pos,1),1))
% axis square
% gamma = h^2;
% contour(abs(F).^2,[gamma gamma],'--w','LineWidth',1.25)


%% Spectrogram and zeros.
figure()
% subplot(1,2,1)
imagesc(-log(abs(F))); hold on;
plot(zeros_pos(:,2),zeros_pos(:,1),'o','Color','w','MarkerFaceColor','w','MarkerSize',2);
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
axis square

%% Classified zeros.
figure()
% imagesc(-log(abs(F))); hold on;
imagesc(abs(F)); hold on;
% triplot(TRIselected,ceros(:,2),ceros(:,1),'c','LineWidth',0.5);
colores = string({'c';'green';'red'});
symbols = string({'o';'^';'d'});
for i = 1:3
    plot(zeros_pos(class==i,2),...
        zeros_pos(class==i,1),symbols(i),...
        'Color',colores(i),'MarkerFaceColor',colores(i),'MarkerSize',2);
end
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
% title('Log-Spectrogram and zeros','Interpreter','latex')
% colormap pink
% axis square

if save_figures
    print_figure('figures/mixture.pdf',8.3,4,'RemoveMargin',true)
end

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
xlabel('$\Vert G_{z} \Vert_{1}$','Interpreter','latex');
ylabel('$H_{\infty}(G_{z})$','Interpreter','latex');
xticklabels([]); yticklabels([])
xticks([]); yticks([])
legend('boxoff');
legend('Location','southwest');

if save_figures
print_figure('figures/feature_space_mixutre.pdf',7,5,'RemoveMargin',true)
end
