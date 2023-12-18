% script load_signal
%
% generate a test signal s of length N, using variable signal = 1-6
%
% Tests signals:
%
%
% Recommended requirements: TFTB (http://tftb.nongnu.org/index_fr.html)
%
% Author: D. Fourer (dominique@fourer.fr)
% Date: Jan 2016
%
%

if ~exist('N', 'var')
  N = 1000;    
end

if ~exist('fmlin')
     error('Please install TFTB (http://tftb.nongnu.org/index_fr.html)')
end
     
if (signal == 1)  %% 3 components
 % the components :
 S = zeros(3, N);
 S(1, :) = real(fmlin(N, 0.1,0.2));
 
 S(2, :) = 2*real(fmlin(N,0.15,0.25));
 
 S(3, :) = 0.5*real(fmlin(N,0.25,0.35));
 S(3, 1:300) = 0; S(3, 701:1000) = 0;

elseif signal == 2
  S = zeros(3, N); 
  
  S(1, :)        = real(fmlin(N,0.1,0.2)) ;
  S(1, 801:1000) = zeros(200,1) ;

  S(2, :)        = 2 * real(fmlin(N,0.15,0.25)) ;
  S(2, 1:200)    = zeros(200,1) ;

  S(3, :)        = 0.5*real(fmlin(N,0.25,0.35));
  S(3, 1:300)    = zeros(300,1);
  S(3, 701:1000) = zeros(300,1);
elseif signal == 3
 S = zeros(2, N);
 S(1, :) = real(fmconst(N, 0.1));
 
 S(2, :) = 1.5*real(fmlin(N,0.15,0.25));
elseif signal == 4
  S = zeros(4, N);
  
  S(1,:) = real(fmconst(N, 0.07));
  
  N_chirp = round(N/2);
  pos_chirp = [1 200 400];
  amp_chirp = [0.5 0.7 0.9];
  for i = 1:3
   n = pos_chirp(i):(pos_chirp(i)+N_chirp-1);
   
   S(1+i, n) = amp_chirp(i)*real(fmlin(N_chirp,0.3,0.45));
  end
  
  %S(5,:) = 1.2*real(fmsin(N, 0.10, 0.25, N/3)); %period,t0,fnorm0,pm1
    
end

s= sum(S);
s = s(:);