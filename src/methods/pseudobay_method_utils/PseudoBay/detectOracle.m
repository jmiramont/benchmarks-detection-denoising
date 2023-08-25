function [MD] = detectOracle(tfr,M,L,tf)



data = transpose(abs(tfr(1:round(M/2),:))).^2 + eps; % Absolute value of the top half TFR
% data = transpose(tfr(1:round(M/2),:)); % Absolute value of the top half TFR

[Niter,N] = size(data); % Extract dimenssions

[F_mat]=compForacle_STFT(M,L);

%% Estimation noise mean
LBcgk = estimB(transpose(data),tf);


%% Estimation amplitude
A_hat = estim_amplitude(data,tf,LBcgk);
A_hat = sqrt(A_hat); 
A_hat(tf==1) = 0;

% LBcgk = LBcgk * N;

%% Detection for MCS
p0 = zeros(Niter,1);      % Initialization p0
p1=p0;                    % Initialization p1

%% Matrix precomputation
w_int = linspace(1e-4,1-1e-4,200);
% w_int = logspace(-6,0,500);
cdw = (1./(w_int+eps))-1;
Lc = length(cdw);
cdw = reshape(cdw,1,1,Lc);
Ftl=(cdw.*F_mat);
cdw = squeeze(cdw);
MD = zeros(Niter,1);

%% Component wise detection
P_d = 0.5;
%% Forward estimation
for t=1:Niter
    Y=data(t,:); % load current frame
    [P_d,p0(t),p1(t)] = detect_QL4(Y,P_d,LBcgk,A_hat(t),cdw,Ftl(:,tf(t),:));
%         [P_d,p0(t),p1(t)] = detect_QL4(Y,P_d,LBcgk,A_hat(t,Nc),cdw,Ftl);
    P_d = min(max(P_d,1e-3),1-1e-3);
    MD(t) = P_d>0.5;

end


%     figure(3)
%     subplot(2,1,1)     
%     plot(A_hat)
%     title('A')
%     subplot(2,1,2)
%     plot(MD)
%     title('detec')
%     ylim([-0.25 1.25])


% 
%     figure
%     subplot(4,1,1)     
%     plot(A_hat)
%     title('A')
%     subplot(4,1,2)
%     plot(p0)
%     title('p0')
%     subplot(4,1,3)
%     plot(p1)
%     title('p1')
%     subplot(4,1,4)
%     plot(MD)
%     title('detec')
%     ylim([-0.25 1.25])
