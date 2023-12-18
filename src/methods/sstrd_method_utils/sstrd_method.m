function xr = sstrd_method(x, Ncomp, use_sst, Pnei, M, L)
N = length(x);

% Default parameters
if nargin <3 || isempty(use_sst)
    use_sst=true;
end

if nargin<5 || isempty(M)
    M=2*N;
end

if nargin<6 || isempty(L)
    L=ceil(sqrt(M)/sqrt(2*pi)/2);
    % disp(L)
end


if nargin<4 || isempty(Pnei)
    Pnei = round(3*M/2/pi/L); % One std dev in freq.
    % disp(Pnei)
end


[tfr,stfr]  = tfrsgab2(x, M, L);
reconstruct_func = @(atfr,L,M)rectfrgab(atfr, L, M);

% Use synchrosqueezed version
if use_sst
    tfr = stfr;
    reconstruct_func = @(atfr,L,M)rectfrsgab(atfr, L, M);
end

% Apply the method
[~, mask] = Brevdo_modeExtract(tfr, L, Ncomp, Pnei);


% Generate a combined mask of all components.
mask_total = sum(mask,3);
mask_total(mask_total~=0) = 1;

% Recover components and signal
x_hat = zeros(N,Ncomp);

% Inversion of the masked STFT.
for c = 1:Ncomp
    x_hat(:,c) = real(reconstruct_func(tfr .* mask(:,:,c), L, M));
    %     [xr,~] = tfristft(tfr .* mask(:,:,c),1:N,w,0);
end

% Return reconstructed signal.
xr = real(reconstruct_func(tfr .* mask_total, L, M));