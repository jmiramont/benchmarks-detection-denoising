function [f_rec, AttenFactorMap, flag_depth] = BlockThresholding(fn, time_win, f_sampling, sigma_noise)

if nargin < 2
N = length(fn);
time_win = round(N/8);
end

if nargin < 3
N = length(fn);
f_sampling = N;
end

if nargin < 4
sigma_noise = 0.5;
end



% disp(time_win)
% disp(f_sampling)
% disp(sigma_noise)

% Block attenuation with bi-dimensional (time and frequency, LxW) blocks. 
% Block size selected by SURE from LxW to 2^(-Kl+1)L x 2^(-Kw+1)W 
%
% The blocks in each Macroblock are of the same size. The Macroblock is of
% size Lmacro x Wmacro, with Lmacro = Cl x Lmax and Wmacro = Cw x Wmax. We
% take Cl = 1, Cw > 1, i.e., the Macroblock is in vertical sense. 
% 

% Maximum block length and width
Lmax = 8;
Wmax = 16;
% Block length = Lmax*2^(-kl), kl=0,...,Kl
Kl = 3;
% Block width = Wmax*2^(-kw), kw=0,...,Kw
Kw = 5;
% Macroblock is of size Lmax x Wmax
Cw = 1;
Wmacro = Wmax * Cw;

% STFT
factor_redund = 1;
STFTcoef = STFT(fn, time_win, factor_redund, f_sampling);

AttenFactorMap = zeros(size(STFTcoef));
flag_depth = zeros(size(STFTcoef));

STFTcoef_th = zeros(size(STFTcoef));

sigma_noise_hanning = sigma_noise * sqrt(0.375);

nb_Macroblk = floor(size(STFTcoef, 2) / Lmax);  

half_nb_Macroblk_freq = floor(((size(STFTcoef, 1)-1)/2)/ Wmacro); 

% A matrix that stores the lambda for different LxW  configurations. 
% (8x16), (8x8), (8x4), (8x2), (8x1)  
% (4x16), (4x8), (4x4), (4x2), (4x1)
% (2x16)  (2x8), (2x4), (2x2), (2x1)
M_lambda = zeros(Kl, Kw);
M_lambda = [1.5, 1.8, 2, 2.5, 2.5;
            1.8, 2,  2.5, 3.5, 3.5;
             2, 2.5, 3.5,  4.7,  4.7];

for i = 1 : nb_Macroblk
    
    % Note that the size of the Hanning window is ODD. We have both -pi, pi
    % and 0 frequency components. 
    % Use L = Lmax = 8, lambda = 2.5 for components at zero frequency. 
    L_pi = 8;
    lambda_pi = 2.5;
    a = 1 - lambda_pi*L_pi*(sigma_noise_hanning)^2*size(STFTcoef, 1) ./ sum(abs(STFTcoef(1, (i-1)*Lmax+1:(i-1)*Lmax + L_pi).^2));
    a = a * (a>0);
    STFTcoef_th(1, (i-1)*Lmax+1:(i-1)*Lmax + L_pi) = a .* STFTcoef(1, (i-1)*Lmax+1:(i-1)*Lmax + L_pi);
    AttenFactorMap(1, (i-1)*Lmax+1:(i-1)*Lmax + L_pi) = a;
    flag_depth(1, (i-1)*Lmax+1:(i-1)*Lmax + L_pi) = 13;
    
    % For negative frequencies
    for j = 1 : half_nb_Macroblk_freq
        SURE_M = zeros(Kl, Kw);
            
        % loop over block length in time
        for klkl = 1 : Kl
            ll = Lmax * 2^(-klkl+1);
            % loop over block width in frequency
            for kwkw = 1 : Kw
                ww = Wmax * 2^(-kwkw+1);
                lambda_JS = M_lambda(klkl, kwkw);

                % loop over blocks in time
                for ii = 1 : 2^(klkl-1);
                    % loop over blocks in frequency.
                    % Note that for Macroblock purpose, Cw is taken into
                    % account. 
                    for jj = 1 : 2^(kwkw-1) * Cw
                        B = STFTcoef(1+(j-1)*Wmacro+(jj-1)*ww+1:1+(j-1)*Wmacro+jj*ww, (i-1)*Lmax+(ii-1)*ll+1:(i-1)*Lmax+ii*ll);

                        % Normalize the coeffcients for the real/imaginary part has unity noise
                        % variance.
                        % 1/sqrt(2) for normalizing real and imaginary
                        % parts separately.
                        B_norm = B / (sqrt(size(STFTcoef, 1))*sigma_noise_hanning/sqrt(2));
                        size_blk = ll * ww;

                        S_real = sum(real(B_norm(:)).^2);
                        SURE_real = size_blk + (lambda_JS^2*size_blk^2-2*lambda_JS*size_blk*(size_blk-2))/S_real*(S_real>lambda_JS*size_blk) + (S_real-2*size_blk)*(S_real<=lambda_JS*size_blk);

                        SURE_M(klkl, kwkw) = SURE_M(klkl, kwkw) + SURE_real;                      
                    end
                end
            end
            
            % Get the configuration that has the minimum error
            [min_error, idx_min] = min(SURE_M(:));
            [klkl, kwkw] = ind2sub([Kl, Kw], idx_min);
        end
        

        % Do the block segmentation and attenuation with the configuration 
        % that has the minimum error
        ll = Lmax * 2^(-klkl+1);
        ww = Wmax * 2^(-kwkw+1); 
        lambda_JS = M_lambda(klkl, kwkw);
        for ii = 1 : 2^(klkl-1);
            % loop over block in frequency
            % Note that for Macroblock purpose, Cw is taken into
            % account.
            for jj = 1 : 2^(kwkw-1) * Cw
                B = STFTcoef(1+(j-1)*Wmacro+(jj-1)*ww+1:1+(j-1)*Wmacro+jj*ww, (i-1)*Lmax+(ii-1)*ll+1:(i-1)*Lmax+ii*ll);
                a = (1 - lambda_JS*ll*ww*(sigma_noise_hanning)^2*size(STFTcoef, 1) ./ sum(abs(B(:)).^2));
                a = a * (a > 0);

                STFTcoef_th(1+(j-1)*Wmacro+(jj-1)*ww+1:1+(j-1)*Wmacro+jj*ww, (i-1)*Lmax+(ii-1)*ll+1:(i-1)*Lmax+ii*ll) = a .* B;
                AttenFactorMap(1+(j-1)*Wmacro+(jj-1)*ww+1:1+(j-1)*Wmacro+jj*ww, (i-1)*Lmax+(ii-1)*ll+1:(i-1)*Lmax+ii*ll) = a;
                flag_depth(1+(j-1)*Wmacro+(jj-1)*ww+1:1+(j-1)*Wmacro+jj*ww, (i-1)*Lmax+(ii-1)*ll+1:(i-1)*Lmax+ii*ll) = idx_min;
            end
        end
    end
    
    
    % For the last few frequencies that do not make a 2D Macroblock, do BlockJS
    % with 1D block.  % Use L = 8, lambda = 2.5 for these frequencies. 
    L_pi = 8;
    lambda_pi = 2.5;
    if 1+Wmacro*half_nb_Macroblk_freq+1 <= (size(STFTcoef, 1)+1)/2
        a_V = (1 - lambda_pi*L_pi*(sigma_noise_hanning)^2*size(STFTcoef, 1) ./ sum(abs(STFTcoef(1+Wmacro*half_nb_Macroblk_freq+1:(end+1)/2, (i-1)*Lmax+1:(i-1)*Lmax + L_pi)).^2,2));
        a_V = a_V .* (a_V > 0);

        STFTcoef_th(1+Wmacro*half_nb_Macroblk_freq+1:(end+1)/2, (i-1)*Lmax+1:(i-1)*Lmax + L_pi) = repmat(a_V, [1, L_pi]) .* STFTcoef(1+Wmacro*half_nb_Macroblk_freq+1:(end+1)/2, (i-1)*Lmax+1:(i-1)*Lmax + L_pi);
        AttenFactorMap(1+Wmacro*half_nb_Macroblk_freq+1:(end+1)/2, (i-1)*Lmax+1:(i-1)*Lmax + L_pi) = repmat(a_V, [1, L_pi]);
        flag_depth(1+Wmacro*half_nb_Macroblk_freq+1:(end+1)/2, (i-1)*Lmax+1:(i-1)*Lmax + L_pi) = 13;
    end
   
end

% For positive frequencies, conjugate from the negative frequencies
STFTcoef_th(size(STFTcoef,1):-1:(end+1)/2+1, :) = conj(STFTcoef_th(2:(end+1)/2, :));
AttenFactorMap(size(STFTcoef,1):-1:(end+1)/2+1, :) =  AttenFactorMap(2:(end+1)/2, :);
flag_depth(size(STFTcoef,1):-1:(end+1)/2+1, :) = flag_depth(2:(end+1)/2, :);
    
% For the last few coefficients that do not make up a block, do hard
% thresholding
STFTcoef_th(:, nb_Macroblk*Lmax+1 : end) = STFTcoef(:, nb_Macroblk*Lmax+1 : end) .* (abs(STFTcoef(:, nb_Macroblk*Lmax+1 : end)) / sqrt(size(STFTcoef, 1)) > 3 * sigma_noise_hanning);

% Inverse Windowed Fourier Transform
f_rec = inverseSTFT(STFTcoef_th, time_win, factor_redund, f_sampling, length(fn));

% Empirical Wiener
STFTcoef_ideal = STFT(f_rec, time_win, factor_redund, f_sampling);
STFTcoef_wiener = zeros(size(STFTcoef));
% % Wiener (Note that the noise' standard deviation is 1/(sqrt(2)) times
% % after multiplying with a Hanning window.)
STFTcoef_wiener = STFTcoef .* (abs(STFTcoef_ideal).^2 ./ (abs(STFTcoef_ideal).^2 + size(STFTcoef, 1)*(sigma_noise_hanning)^2));
% Inverse Windowed Fourier Transform
f_rec = inverseSTFT(STFTcoef_wiener, time_win, factor_redund, f_sampling, length(fn));
%disp(f_rec);



function STFTcoef = STFT(f, time_win, factor_redund, f_sampling)
%
% 1D Windowed Fourier Transform. 
%
% Input:
% - f: Input 1D signal.
% - time_win: window size in time (in millisecond).
% - factor_redund: logarithmic redundancy factor. The actual redundancy
%   factor is 2^factor_redund. When factor_redund=1, it is the minimum
%   twice redundancy. 
% - f_sampling: the signal sampling frequency in Hz.
%
% Output:
% - STFTcoef: Spectrogram. Column: frequency axis from -pi to pi. Row: time
%   axis. 
%
% Remarks:
% 1. The last few samples at the end of the signals that do not compose a complete
%    window are ignored in the transform in this Version 1. 
% 2. Note that the reconstruction will not be exact at the beginning and
%    the end of, each of half window size. However, the reconstructed
%    signal will be of the same length as the original signal. 
%
% See also:
% inverseSTFT
%
% Guoshen Yu
% Version 1, Sept 15, 2006


% Check that f is 1D
if length(size(f)) ~= 2 | (size(f,1)~=1 && size(f,2)~=1)
    error('The input signal must 1D.');
end

if size(f,2) == 1
    f = f';
end

% Window size
% disp(['time win' string(time_win/1000) 'f_sampling' string(f_sampling)]);
size_win = round(time_win/1000 * f_sampling);




% Odd size for MakeHanning
if mod(size_win, 2) == 0
    size_win = size_win + 1;
end
halfsize_win =  (size_win - 1) / 2;

w_hanning = MakeHanning(size_win); 

Nb_win = floor(length(f) / size_win * 2);

% STFTcoef = zeros(2^(factor_redund-1), size_win, Nb_win-1);
STFTcoef = zeros(size_win, (2^(factor_redund-1) * Nb_win-1));

shift_k = round(halfsize_win / 2^(factor_redund-1));
% Loop over 
for k = 1 : 2^(factor_redund-1)    
    % Loop over windows
    for j = 1 : Nb_win - 2 % Ingore the last few coefficients that do not make a window
        f_win = f(shift_k*(k-1)+(j-1)*halfsize_win+1 : shift_k*(k-1)+(j-1)*halfsize_win+size_win);
        STFTcoef(:, (k-1)+2^(factor_redund-1)*j) = fft(f_win .* w_hanning');
    end
end


function f_rec = inverseSTFT(STFTcoef, time_win, factor_redund, f_sampling, length_f)
%
% Inverse windowed Fourier transform. 
%
% Input:
% - STFTcoef: Spectrogram. Column: frequency axis from -pi to pi. Row: time
%   axis. (Output of STFT). 
% - time_win: window size in time (in millisecond).
% - factor_redund: logarithmic redundancy factor. The actual redundancy
%   factor is 2^factor_redund. When factor_redund=1, it is the minimum
%   twice redundancy. 
% - f_sampling: the signal sampling frequency in Hz.
% - length_f: length of the signal. 
%
% Output:
% - f_rec: reconstructed signal. 
%
% Remarks:
% 1. The last few samples at the end of the signals that do not compose a complete
%    window are ignored in the forward transform STFT of Version 1. 
% 2. Note that the reconstruction will not be exact at the beginning and
%    the end of, each of half window size. 
%
% See also:
% STFT
%
% Guoshen Yu
% Version 1, Sept 15, 2006

% Window size
size_win = round(time_win/1000 * f_sampling);

% Odd size for MakeHanning
if mod(size_win, 2) == 0
    size_win = size_win + 1;
end
halfsize_win =  (size_win - 1) / 2;

Nb_win = floor(length_f / size_win * 2);

% Reconstruction
f_rec = zeros(1, length_f);

shift_k = round(halfsize_win / 2^(factor_redund-1));

% Loop over windows 
for k = 1 : 2^(factor_redund-1)
    for j = 1 : Nb_win - 1
        f_win_rec = ifft(STFTcoef(:, (k-1)+2^(factor_redund-1)*j));
        f_rec(shift_k*(k-1)+(j-1)*halfsize_win+1 : shift_k*(k-1)+(j-1)*halfsize_win+size_win) =  f_rec(shift_k*(k-1)+(j-1)*halfsize_win+1 : shift_k*(k-1)+(j-1)*halfsize_win+size_win) +  (f_win_rec');
    end
end

f_rec = f_rec / 2^(factor_redund-1);