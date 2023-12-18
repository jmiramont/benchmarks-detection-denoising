function [signal_r,mask] = ...
 szc_method(signal,noise_std,J,criterion,margins,beta,Nfft)
% Filters noise based on the classification of the spectrogram zeros and
% the Delaunay triangulation. First, the 2D histogram of zeros is computed.
% Then the zeros are classified by using a unsupervised strategy to detect
% those related only due to noise-noise interference and separate them from
% those produced by the signal and noise.
%
% To use this function, you must have the Time-Frequency Toolbox developed 
% by François Auger, Olivier Lemoine, Paulo Gonçalvès and Patrick Flandrin
% in Matlab's path variable. 
% You can get a copy of the toolbox from: http://tftb.nongnu.org/
%
% Other functions needed:
% - roundgauss()
% - find_spectrogram_zeros()
% - compute_zeros_histogram()
% - classify_spectrogram_zeros()
% - compute_centroids()
%
% Input:
% - signal:     Signal to process (must be a vector, either column or row).
% - noise_std:  Standard deviation of the noise realizations used to 
%               compute the 2D histogram of zeros. If 'estimate' is passed
%               insted of a float, the parameter is estimated from the
%               signal.
% - J:          Number of noise realizations used to compute the 2D
%               histogram. Defaults to 512.
% - criterion:  A two-elements cell with a combination of and clustering 
%               algorithm ('gmm' or 'knn') and a number of clusters 
%               criterion ('gap' or 'ch'). Examples:
%               -   {'gmm','gap'}
%               -   {'knn','ch'}
% 
% - margins:    A 2x1 vector. When computing the zeros of the spectrogram, 
%               only consider those zeros far from the border of the plane.
% - beta:       A constant that multiplies the estimated noise_std. Use 
%               this option when you need to modify the noise estimation.
%
% Output:
% - mask:       Extracting mask to filter the short-time Fourier transform.
% - signal_r:   Denoised signal.
% - TRI:        Delaunay triangulation on the spectrogram zeros.
% - TRIse:       Selected triangles from the Delaunay triangulation.
% - zeros_pos:  A [N,2] array with the time-frequency coordenates of the
%               zeros of the spectrogram. Where N is the number of zeros.
% - class:      A [N,1] vector with assigned kind of zeros (1,2 or 3).
% - K:          Number of clusters detected. K=1 means only noise. K=2
%               means a signal is present. K=3 means that zeros of
%               interference between components are present.
% - features:   A [N,2] array with the values of the features computed
%               for each zero.
% - zeros_hist: A matrix containing the 2D histogram of the spectrogram
%               zeros.
%
% Example:
%         N = 2^8;
%         x = real(fmlin(N,0.10,0.25)+fmlin(N,0.15,0.3)).*tukeywin(N,0.4);
%         xn = sigmerge(x,randn(size(x)),20);
%         [mask, xr] = classified_zeros_denoising(xn);
%         figure(); 
%         subplot(121); plot(x,'b','DisplayName','Signal'); hold on; 
%         plot(xr,'--r','DisplayName','Denoised Signal'); legend();
%         subplot(122); imagesc(mask); title('Extraction Mask');
%
% September 2023
% Author: Juan M. Miramont-Taurel <juan.miramont@univ-nantes.fr>
% -------------------------------------------------------------------------

N = length(signal);

if isrow(signal)
    signal = transpose(signal);
end

if nargin<2 || isempty(noise_std)
    noise_std = 'estimate';
end

if nargin<3 || isempty(J)
    J = round(N/4); % Number of noise realizations.
end

if nargin<4 || isempty(criterion)
    criterion = {'gmm','gap'};
end

if nargin<5 || isempty(margins)
    margins = [0,0]; %
end

if nargin<6 || isempty(beta)
    beta = 1;
end

if nargin<7 || isempty(Nfft)
    Nfft = 2*N;
end

% Compute the original zeros, the 2D histogram, the STFT, spectrogram, etc.
% tic()
disp(Nfft)
[zeros_hist, zeros_pos, F, w, T, N, S] = compute_zeros_histogram(signal,...
                                                              noise_std,...
                                                              J,...
                                                              margins,...
                                                              beta,...
                                                              Nfft);
% toc()

zeros_hist(1,:) = 0; zeros_hist(end,:) = 0;
zeros_hist(:,1) = 0; zeros_hist(:,end) = 0;

% zeros_hist = imgaussfilt(zeros_hist);
% figure(); imagesc(log(zeros_hist2+eps));
% figure(); imagesc(log(zeros_hist+eps));

if nargin < 4
    r = 3*ceil(T/8); % Radius of the histograms vicinity.
end


% Classify the zeros based on the histogram:
% tic()
[class, K, features,zeros_pos] = classify_spectrogram_zeros_2(zeros_hist,...
                                                              zeros_pos,...
                                                              J,...
                                                              [2,3],...
                                                              criterion,...
                                                              false);
% toc()


% margin_row = margins(1); margin_col = margins(2);
% invalid_ceros = zeros(length(zeros_pos),1);
% invalid_ceros(zeros_pos(:,1)<margin_row | zeros_pos(:,1)>(size(S,1)-margin_row))=1;
% invalid_ceros(zeros_pos(:,2)<margin_col | zeros_pos(:,2)>(size(S,2)-margin_col))=1;
% invalid_ceros = logical(invalid_ceros);
% valid_ceros = ~invalid_ceros;
% % number_of_valid_ceros = sum(valid_ceros);
% zeros_pos=zeros_pos(valid_ceros,:);



% % Compute the triangulation and the mask based on the classified zeros.
%     TRI = delaunay(zeros_pos);
%     for i = 1:size(TRI,1)
% %    Disrupt triangles with just one NN zero.
%         if sum(class(TRI(i,:))==2)==1 
%             zeros_pos(TRI(i,class(TRI(i,:))==2),:) = 1; 
%         end
%     end


    TRI = delaunay(zeros_pos);
    criteria_1 = zeros(size(TRI,1),1);
    criteria_2 = zeros(size(TRI,1),1);
    for i = 1:size(TRI,1)
% %         Criteria 1: Triangles with at least one zero of the first kind.
        if any(class(TRI(i,:))==1)
            criteria_1(i) = 1;
        end

% %         Criteria 2: Triangles with all zeros of the third kind.
%         class(class==1) = 3;
        if all(class(TRI(i,:))==3)
            criteria_2(i) = 1;
        end
    end


    % Select triangles satisfying either criteria:
if K>1    
    TRIse = TRI(criteria_1 | criteria_2,:);
else    
    TRIse = TRI;
end

mask = mask_from_triangles(S,TRIse,zeros_pos);

% If the signal is real, reflect the mask before inversion.
if size(mask,1)<size(F,1)
    aux = zeros(size(F));
    aux(1:size(mask,1),:) = mask;
    aux(size(mask,1)+1:end,:) = mask(end-1:-1:2,:);
    mask = aux;
end

% Inversion of the masked STFT.
[signal_r,~] = tfristft(F.*mask,1:N,w,0);
signal_r = signal_r.';