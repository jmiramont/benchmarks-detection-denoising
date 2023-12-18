clear all; close all;

save_figures = false;

rng(0);
N = 2^9;
Nchirp = N-100;
tmin = round((N-Nchirp)/2);
tmax = tmin + Nchirp;
x = zeros(N,1);
tchirp = (0:Nchirp-1);
instf = 0.1+0.25*tchirp/Nchirp + 0.1*sin(2*pi*tchirp/Nchirp);
xsub = cos(2*pi*cumsum(instf)).'.*tukeywin(Nchirp,0.25);
x(tmin+1:tmax) = xsub;

% Parameters for the STFT.
Nfft = 2*N;
fmax = 0.5; % Max. norm. frequency to compute the STFT.
[w,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.


[F,~,~] = tfrstft(x,1:N,Nfft,w,0);
F = F(1:floor(Nfft*fmax),:);
F = flipud(F);
S = abs(F).^2;


% Noise realization:
SNRin = 15;
noise = randn(size(x));
noise = noise*sqrt(10^(-SNRin/10)*sum(x.^2)/N);
xnoise = x+noise;
gamma = var(noise);
[Fnoise,~,~] = tfrstft(noise,1:N,Nfft,w,0);
Fnoise = Fnoise(1:floor(Nfft*fmax),:);
Fnoise = flipud(Fnoise);
Snoise = abs(Fnoise).^2;

Fmix = F+Fnoise;
Smix = abs(Fmix).^2;

SNR = 10*log10(sum(x.^2)/sum(noise.^2));


%%
figure()
plot(x,'k','LineWidth',0.75); hold on;
% plot(x+noise,'k')
xlim([tmin,tmax])
ylim([-1.15,1.15])
xticklabels([])
yticklabels([])
xticks([])
yticks([])
xlabel('time', 'Interpreter','latex');
ylabel('amplitude', 'Interpreter','latex');
title('Signal', 'Interpreter','latex')
if save_figures
print_figure('figures/signal_example.pdf',7.5,4.2,'RemoveMargin',true)
end

%%
figure()
% subplot(1,3,3)
% imagesc(-log(abs(Fmix))); hold on;
Fmix = Fmix/sum(Fmix(:));
imagesc((abs(Fmix))); hold on;
ceros2 = find_spectrogram_zeros(Smix);
plot(ceros2(:,2),ceros2(:,1),'o','Color','w','MarkerFaceColor','w','MarkerSize',1.0);
contour(S,[gamma gamma],'--r','LineWidth',1.5)
% title('Noise + Signal','Interpreter','latex');
% colormap pink
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); 
ylabel('frequency','Interpreter','latex')
% axis square;

if save_figures
print_figure('figures/level_set_mix.pdf',4.2,4.2,'RemoveMargin',true)
end

Fmix = Fmix/sum(Fmix(:));

%%
figure()
% subplot(1,3,3)
% imagesc(-log(abs(Fmix))); hold on;
Fmix = Fmix/sum(Fmix(:));
imagesc((abs(Fmix))); hold on;
ceros2 = find_spectrogram_zeros(Smix);
plot(ceros2(:,2),ceros2(:,1),'o','Color','w','MarkerFaceColor','w','MarkerSize',1.0);
% contour(S,[gamma gamma],'--r','LineWidth',1.5)
title('Noise + Signal','Interpreter','latex');
% colormap pink
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); 
%ylabel('frequency','Interpreter','latex')
% axis square;

if save_figures
print_figure('figures/level_set_mix_no_gamma.pdf',3.6,4.2,'RemoveMargin',true)
end

Fmix = Fmix/sum(Fmix(:));

%%
hf = figure('Units','normalized');
subplot(1,2,2)
% colormap gray
hCB = colorbar('west');
set(gca,'Visible',false)
hCB.Position = [0.6 0.15 0.2 0.74];
hCB.FontSize = 6;
hCB.TickLabelInterpreter ='latex';
hf.Position(4) = 0.1000;
clim([min((abs(Fmix(:)))),max((abs(Fmix(:))))]);
% print_figure('colorbar_spect_1.pdf',0.99,4.2,'RemoveMargin',true);


%%
figure()
% imagesc(abs(F)); hold on;
% F = F/sum(abs(F(:)));
imagesc(abs(F)); hold on;
ceros2 = find_spectrogram_zeros(abs(F));
plot(ceros2(:,2),ceros2(:,1),'o','Color','w','MarkerFaceColor','w','MarkerSize',1.0);
% contour(S,[gamma gamma],'--r')
title('Signal','Interpreter','latex');
% colormap pink
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
axis square;

% if save_figures
% print_figure('figures/level_set_signal.pdf',4.0,4.2,'RemoveMargin',true)
% end





%%
figure()
% subplot(1,3,2)
% imagesc(-log(abs(Fnoise)));
Fnoise = Fnoise/sum(abs(Fnoise(:)));
imagesc((abs(Fnoise)));
ceros1 = find_spectrogram_zeros(Snoise); hold on;
plot(ceros1(:,2),ceros1(:,1),'o','Color','w','MarkerFaceColor','w','MarkerSize',1.0);
title('Noise','Interpreter','latex');
% colormap pink
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
% axis square;


if save_figures
print_figure('figures/level_set_noise.pdf',4.0,4.2,'RemoveMargin',true)
end




%% Bar

hf = figure('Units','normalized');
subplot(1,2,2)
% colormap gray
hCB = colorbar('west');
set(gca,'Visible',false)
hCB.Position = [0.4 0.15 0.2 0.74];
hCB.FontSize = 6;
hCB.TickLabelInterpreter ='latex';
hf.Position(4) = 0.1000;
clim([min((abs(Fnoise(:)))),max((abs(Fnoise(:))))]);
% print_figure('colorbar_spect_2.pdf',0.95,4.2,'RemoveMargin',true);