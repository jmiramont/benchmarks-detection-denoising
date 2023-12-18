function [signal_r, mask,ceros,TRIselected,S] = dt_method(signal, LB, Nfft,L)

N = length(signal);

if nargin<3
    Nfft = 2*N;
end

if nargin<4
    L = sqrt(Nfft);
end

l=floor(sqrt(-Nfft*log(1e-15)/pi))+1;
w=amgauss(2*l+1,l+1,L);
w=w/norm(w);
Ly = Nfft/L;
margins = [Ly/2,L/2];


% Get the spectrogram.
% [w,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.

[F,~,~] = tfrstft(signal,1:N,Nfft,w,0);
F = F(1:Nfft/2+1,:);
F = flipud(F);
S = abs(F).^2;

% Find original zeros
ceros = find_spectrogram_zeros(S);

% Keep zeros within margins:
margin_row = 2; margin_col = 2;
invalid_ceros = zeros(length(ceros),1);
invalid_ceros(ceros(:,1)<margin_row | ceros(:,1)>(size(S,1)-margin_row))=1;
invalid_ceros(ceros(:,2)<margin_col | ceros(:,2)>(size(S,2)-margin_col))=1;
invalid_ceros = logical(invalid_ceros);
valid_ceros = ~invalid_ceros;

% Triangulation of zeros
u=ceros(:,1)/Ly;
v=ceros(:,2)/L;
TRI = delaunay(u,v);
TRI2 =  [];

% Keep triangles within the specified margins.
for j = 1:size(TRI,1)
    if ~any(invalid_ceros(TRI(j,:)))
        TRI2 = [TRI2; TRI(j,:)];
    end
end

% Find edge lengths of all triangles.
[~,MAX_EDGES,TRI_EDGES] = describe_triangles(TRI2,ceros,Nfft,L);

% Select triangles based on the length
select_criterion=zeros(size(TRI2,1),1);
UB = 3.0;

% Keep the triangles with an edge larger than lmax
criterion_2 = zeros(size(TRI2,1),1);
for i =1:size(TRI2,1)
    if any(TRI_EDGES(i,:)>LB) && all(TRI_EDGES(i,:)<UB)
        select_criterion(i)=1;
    end
end
TRIselected=TRI2(logical(select_criterion),:);

% Get a 0/1 mask based on the selected triangles.
mask = mask_from_triangles(F,TRIselected,ceros);

% Reconstruction and QRF computation.
signal_r = real(sum(F.*mask))/max(w)/N;

