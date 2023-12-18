clear all; close all;

% Generate a pair of parallel chirps close to each other.
rng(0)
N = 2^8;
Nchirp = N;
tmin = round((N-Nchirp)/2);
tmax = tmin + Nchirp;
x = zeros(N,1);
instf1 = 0.09+0.2*(0:Nchirp-1)/Nchirp;
instf2 = 0.16+0.2*(0:Nchirp-1)/Nchirp;
x(tmin+1:tmax) = (cos(2*pi*cumsum(instf1)) + cos(2*pi*cumsum(instf2))).*tukeywin(Nchirp,0.25).';

% Parameters for the STFT.
Nfft = 2*N;
fmax = 0.5; % Max. norm. frequency to compute the STFT.
[w,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.
% w = tftb_window(2*T+1,'Kaiser');
% w = tftb_window(T+1,'Hanning');
% w = tftb_window(8*2+1,'Gauss');

% Noise realization:
SNRin = 30;
original_noise = randn(size(x));
[xnoise,std_noise] = sigmerge(x,original_noise,SNRin);
[F,~,~] = tfrstft(xnoise,1:N,Nfft,w,0);
F = F(1:floor(Nfft*fmax),:);
F = flipud(F);
S = abs(F).^2;

% Find original zeros and triangulation
tic()
[ceros,Qz] = find_spectrogram_zeros(S);
toc()
TRI = delaunay(ceros);

% Keep zeros within margins:
margin_row = 0; margin_col = 0;
invalid_ceros = zeros(length(ceros),1);
invalid_ceros(ceros(:,1)<margin_row | ceros(:,1)>(size(S,1)-margin_row))=1;
invalid_ceros(ceros(:,2)<margin_col | ceros(:,2)>(size(S,2)-margin_col))=1;
invalid_ceros = logical(invalid_ceros);
valid_ceros = ~invalid_ceros;
% number_of_valid_ceros = sum(valid_ceros);
ceros=ceros(valid_ceros,:);

figure()
% imagesc(log(abs(F))); hold on;
imagesc(abs(F).^2); hold on;
plot(ceros(:,2),ceros(:,1),'wo','MarkerFaceColor','w','MarkerSize',1.5);
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
% Uncomment to save
% print_figure('figures/spectrogram_b.pdf',4.2,4.2,'RemoveMargin',true)


%% Histogram using noise with the same variance as the original.
disp('Computing histogram...');
lims = 0;
aux_plane = zeros(size(S));
M = 2048; % Number of noise realizations.
% SNRalg = SNRin;
% SNRalg = estimate_SNR_2(xnoise);

noise_alg = randn(N,1);
noise_alg = noise_alg/std(noise_alg)*std_noise;
xnoise_alg = xnoise+noise_alg;
%     xnoise_alg = sigmerge(xnoise,noise_alg,SNRalg);
[S_alg,~,~] = tfrsp(xnoise_alg,1:N,Nfft,w,0);
S_alg = S_alg(1:floor(Nfft*fmax),:);
S_alg = flipud(S_alg);
[new_zeros,Qz] = find_spectrogram_zeros(S_alg);
aux_plane = aux_plane + Qz;


ceros(:,1) = N + 1 - ceros(:,1);
new_zeros(:,1) = N + 1 - new_zeros(:,1)  ;

figure()
% imagesc(flipud(-log(abs(F)))); hold on;
plot(ceros(:,2),ceros(:,1),'ko','MarkerSize',4); hold on;
plot(new_zeros(:,2),new_zeros(:,1),'ko','MarkerFaceColor','k','MarkerSize',1.5);
plot(1:N, 2*instf1*N, 'k--');
plot(1:N, 2*instf2*N, 'k--');
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
xlim([2 N-1])
ylim([2 N-1])
% Uncomment to save
% print_figure('figures/position_of_zeros.pdf',4.2,4.2,'RemoveMargin',true)