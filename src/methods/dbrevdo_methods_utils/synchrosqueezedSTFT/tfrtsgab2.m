function [tfr, stfr, lost,lost_v] = tfrtsgab2(x, M, L, gamma_K)
% [tfr] = tfrtsgab(x, M, L, gamma_K)
% Compute the discrete-time time-reassigned synchrosqueezed Gabor Transform
% (using FFT)
% 
% INPUT:
% x      : the signal to process
% M      : number of frequency bins to process (default: length(x))
% L      : window duration parameter:  w0 * T, (default: 10)
% gamma_K: threshold applied on window values (default: 10^(-4))
%
% OUTPUT:
% tfr    : discrete stft
% stfr   : discrete time-reassigned synchrosqueezed stft
%
% Author: D.Fourer
% Date: 28-08-2015
% Ref: [D. Fourer, J. Harmouche, J. Schmitt, T. Oberlin, S. Meignen, F. Auger and P. Flandrin. The ASTRES Toolbox for Mode Extraction of Non-Stationary Multicomponent Signals. Proc. EUSIPCO 2017, Aug. 2017. Kos Island, Greece.]
% Ref: [D. Fourer and F. Auger. Second-order Time-Reassigned Synchrosqueezing Transform: Application to Draupner Wave Analysis. Proc. EUSIPCO 2019, Coruna, Spain.]

x = x(:).';          %convert x as a row vector
N = length(x);

if ~exist('M', 'var')
 M = N;
end
if ~exist('L', 'var')
 L = 10;
end
if ~exist('gamma_K', 'var')
 gamma_K = 10^(-4);
end

lost = 0;
lost_v = zeros(M,1);

tfr    = zeros(M, N);
stfr   = zeros(M, N);

%tfr_d = zeros(M, N);
tfr_t = zeros(M, N);

K = 2 * L * sqrt(2*log(1/gamma_K));  %% window length in samples

A = 1/(sqrt(2*pi)*L);
B = -1i * 2*pi / M;
C = -1 / (2*L^2);

mm = m_axis(M);
for n = 1:N
  
  k_min = min(n-1, round(K/2));
  k_max = min(N-n, round(K/2));

  k = (-k_min):k_max;
  k2 = k.^2;
  g     = A * exp( C * k2);
  %dg    = L^(-2) * k .* g;
  tg    = -k .* g;
  
  tfr(:,n)   = fft(x(n+k) .* g, M) .* exp(B * mm * (n-1 - k_min));
  tfr_t(:,n) = fft(x(n+k) .* tg, M) .* exp(B * mm * (n-1 - k_min));
  
  for m = 1:M
    %nn = n-1;
    %tfr(m,n) = exp(B * mm(m) * nn) * sum( x(n+k) .* g .* exp(B .* mm(m) .* k));  % 
    
    if abs(tfr(m,n)) > eps

     %tfr_d(m,n) = exp(B * mm(m) * nn) * sum( x(n+k) .* dg .* exp(B .* mm(m) .* k));  %exp(B * mm * nn) * 
     %tfr_t(m,n) = exp(B * mm(m) * nn) * sum( x(n+k) .* tg .* exp(B .* mm(m) .* k));  %
     
     n_hat = n - round(              real( tfr_t(m,n) / tfr(m,n) ) );
     %m_hat = m + round( M / (2*pi) *  imag( tfr_d(m,n) / tfr(m,n) ) );
     
      
      %% out of bounds
      %m_out_of_bounds = m_hat < 1 || m_hat > M;
      n_out_of_bounds = n_hat < 1 || n_hat > N;
      
      if n_out_of_bounds %m_out_of_bounds
        lost = lost + abs(tfr(m,n))^2;
        lost_v(m) = lost_v(m) + tfr(m,n); %% keep energy for signal reconstruction
        continue;
      end
      
      stfr(m,n_hat) = stfr(m,n_hat) + tfr(m,n); %/(2*pi) * exp(2*1i*pi*mm(m)*nn/M)
    end
     
  end %% m
end %% n

%% used to obtain matlab conventions
%tfr  = transpose(tfr);
%stfr  = transpose(stfr);
end