function [zeros_hist, zeros_pos, F, w, T, N, S] = compute_zeros_histogram(signal, noise_std, J, margins, beta, Nfft)
% Computes the 2D histogram of the spectrogram zeros of the given signal.
% The histograms are computed by adding J independent white noise
% realizations. The variance of these realizations should be similar to
% that of the noise present in the signal. It can be passed as an argument
% using or estimated using the MAD robust estimator proposed by Donoho.
%
% To use this function, you must have the Time-Frequency Toolbox developed
% by François Auger, Olivier Lemoine, Paulo Gonçalvès and Patrick Flandrin
% in Matlab's path variable.
% You can get a copy of the toolbox from: http://tftb.nongnu.org/
%
% Other functions needed:
% - roundgauss()
% - find_spectrogram_zeros()
%
% Input:
% - signal:     Signal to process (must be a vector, either column or row).
% - noise_std:  Standard deviation of the noise realizations used to
%               compute the 2D histogram of zeros. If 'estimate' is passed
%               insted of a float, the parameter is estimated from the
%               signal.
% - J:          Number of noise realizations used to compute the 2D
%               histogram. Defaults to 512.
% - margins:    A 2x1 vector. When computing the zeros of the spectrogram,
%               only consider those zeros far from the border of the plane.
% - beta:       A constant that multiplies the estimated noise_std. Use
%               this option when you need to modify the noise estimation.
%
% Output:
% - zeros_hist: A matrix containing the 2D histogram of the spectrogram
%               zeros.
% - zeros_pos:  A [N,2] array with the time-frequency coordenates of the
%               zeros of the spectrogram. Where N is the number of zeros.
% - F:          Short-time Fourier transform of the signal.
% - w:          The window used to compute F.
% - T:          The value that parametrizes the width of w.
% - N:          The lenght of the signal.
% - S:          The spectrogram of the signal.
%
% Example:
%         N = 2^9;
%         x = real(fmlin(N,0.10,0.25)+fmlin(N,0.15,0.3)).*tukeywin(N,0.1);
%         xn = sigmerge(x,randn(size(x)),20);
%         [zeros_hist, zeros_pos, F, ~, ~, ~, S] =...
%                              compute_zeros_histogram(xn, 'estimate');
%         zeros_hist = flipud(zeros_hist);S = flipud(S);
%         figure(); subplot(121); imagesc(S); title('Spectrogram');
%         subplot(122); imagesc(log(zeros_hist)); title('2D histogram');
%         colormap jet;
%
% September 2022
% Author: Juan M. Miramont <juan.miramont@univ-nantes.fr>
% -------------------------------------------------------------------------

N = length(signal);


if isrow(signal)
    signal = transpose(signal);
end

if nargin<2 || isempty(noise_std)
    noise_std = 'estimate';
end

if nargin<3 || isempty(J)
    J = 512; % Number of noise realizations.
end

if nargin<4 || isempty(margins)
    margins = [5,5]; %
end

if nargin<5 || isempty(beta)
    beta = 1;
end

if nargin<6 || isempty(Nfft)
    Nfft = 2*N;
end

% Check if the signal is real or complex.
if isreal(signal)
    real_signal_flag = true;
    std_estim_factor = sqrt(2);
else
    real_signal_flag = false;
    std_estim_factor = 1;
end

% Parameters for the STFT.
[w,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.

% Compute STFT and spectrogram.
signal = [signal(N/2+1:-1:2); signal; -signal(N-1:-1:N/2)];
[F,~,~] = tfrstft(signal,N/2+1:N/2+N,Nfft,w,0);
S = abs(F).^2;

% Estimate the noise standard deviation (used later if needed).
absFr = abs(real(F(:)));
stdAlgEst = beta*std_estim_factor*median(absFr)/0.6745;

% Keep only half of the spectrogram if the signal is real.
if real_signal_flag
    S = S(1:round(Nfft/2)+1,:);
end

% Find original zeros.
zeros_pos = find_spectrogram_zeros(S);

% Keep zeros within margins.
margin_row = margins(1); margin_col = margins(2);
invalid_ceros = zeros(length(zeros_pos),1);
invalid_ceros(zeros_pos(:,1)<margin_row | zeros_pos(:,1)>(size(S,1)-margin_row))=1;
invalid_ceros(zeros_pos(:,2)<margin_col | zeros_pos(:,2)>(size(S,2)-margin_col))=1;
invalid_ceros = logical(invalid_ceros);
valid_ceros = ~invalid_ceros;
% number_of_valid_ceros = sum(valid_ceros);
zeros_pos=zeros_pos(valid_ceros,:);

% Compute histogram
disp('Computing histogram...');
lims = 0;
aux_plane = zeros(size(S));
for j  = 1:J
    noise_alg = randn(size(signal));

    if ~real_signal_flag
        noise_alg = noise_alg + 1i*randn(size(signal));
    end

    if noise_std=='estimate'
        noise_alg = (noise_alg-mean(noise_alg))/std(noise_alg)*stdAlgEst;
    else
        noise_alg = (noise_alg-mean(noise_alg))/std(noise_alg)*noise_std;
    end
    xnoise_alg = signal + noise_alg;
    [S_alg,~,~] = tfrsp(xnoise_alg,N/2+1:N/2+N,Nfft,w,0);
    if real_signal_flag
        S_alg = S_alg(1:round(Nfft/2)+1,:);
        %         S  = fftshift(S,1)
        %         S_alg = flipud(S_alg);
    end
    [~, Qz] = find_spectrogram_zeros(S_alg);
    aux_plane = aux_plane + Qz;
end

hist2d = aux_plane(lims+1:end-lims,lims+1:end-lims);
zeros_hist = hist2d;

% disp('Finished.');
