function [MD,P_d]=detect_MCS(data,Ncomp,tempdata,LBcgk,A_hat,F_mat,tf)


%% parameters
[Niter,M] = size(data);
p0 = zeros(Niter,1);      % Initialization p0
p1=p0;                    % Initialization p1

%% Matrix precomputation
w_int = linspace(1e-4,1-1e-4,200);
% w_int = logspace(-6,0,200);
cdw = (1./(w_int+eps))-1;
Lc = length(cdw);
cdw = reshape(cdw,1,1,Lc);
Ftl=(cdw.*F_mat);
cdw = squeeze(cdw);
MD = zeros(Niter,Ncomp);

    
for Nc = 1:Ncomp
    %% Component wise detection
    data = data + tempdata(:,:,Nc); % Current ridge
    P_d = 0.5;
    %% Forward estimation
    for t=1:Niter
        Y=data(t,:); % load current frame
        [P_d,p0(t),p1(t)] = Fdetect(Y,P_d,LBcgk,A_hat(t,Nc),cdw,Ftl(:,tf(t,Nc),:));
        P_d = min(max(P_d,1e-3),1-1e-3);
        MD(t,Nc) = P_d>0.5;
    end
    % Remove current ridge for next component
    data = max(data - tempdata(:,:,Nc), 0);
end