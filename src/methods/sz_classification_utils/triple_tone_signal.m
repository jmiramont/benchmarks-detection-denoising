function [signal, ceros, tones_location] = triple_tone_signal(N)
% Generate a N-sample signal with three tones.
%
% September 2022
% Author: Juan M. Miramont <juan.miramont@univ-nantes.fr>
%--------------------------------------------------------------------------

t = (0:N-1);
Nfft = 2*N;

[w,T] = roundgauss(Nfft,1e-6);

% Impulse positions:
% Distances are aprox > than 3*std(g)
std_g = T/sqrt(2*pi);
Df12 = round(3*std_g);
Df23 = Df12;
f1 = N/2 - Df12;
f2 = f1+Df12;
f3 = f2+Df23;
f1 = f1/Nfft;
f2 = f2/Nfft;
f3 = f3/Nfft;

tones_location = [f1,f2,f3];

% Impulse Amplitudes:
A1 = 1;
fA12 = 1;
fA23 = 1;
A2 = A1*fA12;
A3 = A2*fA23;

% Generate Impulse signal:
x1 = cos(2*pi*f1*t);
x2 = cos(2*pi*f2*t);
x3 = cos(2*pi*f3*t);

% Zeros positions:
fz12 = round(((f1+f2)/2 - T*T*log(fA12)/2/pi/Df12)*Nfft);
fz23 = round(((f3+f2)/2 - T*T*log(fA23)/2/pi/Df23)*Nfft);


tz12 = (((0:N)+0.5)/Df12)*2*N;
tz12 = tz12(tz12>=0 & tz12 < N);
tz23 = (((0:N)+0.5)/Df23)*2*N;
tz23 = tz23(tz23>=0 & tz23 < N);

vz = [ones(size(tz12))*fz12 ones(size(tz23))*fz23].';
uz = [tz12+1 tz23+1].';

ceros = [uz vz];
signal = x1+x2+x3;
signal = signal.';

% Border effects?
% signal = signal.*tukeywin(N,0.);


% [F,~,~] = tfrstft(signal,1:N,Nfft,w,0);
% F = F(1:floor(Nfft*0.5),:);
% tw = zeros(size(F,1),1);
% tw(36:end-35) = tukeywin(size(F,1)-70,0.25);
% 
% for i = 1:size(F,2)
%     F(:,i) = F(:,i).*tw;
% end
% 
% signal = real(sum(F))/max(w);
% signal = signal.';
% 
