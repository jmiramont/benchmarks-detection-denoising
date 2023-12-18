function [MD] = pseudoBay_Oracle(tfr,Ncomp,M,L,div,beta,alpha,ds,Pnei,ifplot,detect,bolOracle,PneiMask,tf0)
%
% Main algorithm: estimate the ridge position and the variance of the
% posterior distribution to propagate information to the next time sequence
%
% INPUT:
% tfr           : Time-frequency represerntation of the MCS
% Ncomp         : Number of component
% M             : Number of frequential bin
% L             : analysis window size (in bin)
% div           : Entropy choice (1 : KL | 2 : beta | 3 : Renyi)
% beta          : beta divergence hyperparameter
% alpha         : Renyi divergence hyperparameter
% ds            : variance of the random walk in the temporal model
% ifplot        : Boolean for debugging
% Pnei          : number of neighbors considered in the mask
%
%
% OUTPUT:
% MD      : Detection Mask


if ~exist('div', 'var')
 div=1;% default KL
end
if (~exist('beta', 'var') && (div==1))
 beta=1;% beta hyperparameter for beta divergence
end
if (~exist('alpha', 'var') && (div==2))
 alpha=0.5;%  alpha hyperparameter for Renyi divergence
end
if ~exist('ds', 'var')
 ds=3;% variance of the random walk in the temporal model
end
if ~exist('Pnei', 'var')
 Pnei=10;% default KL
end

data = transpose(abs(tfr(1:round(M/2),:))).^2 + eps; % Absolute value of the top half TFR
[Niter,N] = size(data); % Extract dimenssions

if bolOracle==0



    [F_mat, Fbeta_mat, IntFbeta, Falpha_mat, ~, F_sAB]=compFbeta_STFT(beta,alpha,M,L,N);
    LFM = log(F_mat+eps);

    data0=data;

    %% Initialization
    tf=zeros(Niter,Ncomp); %Array to store the means of the depth
    stf=zeros(Niter,Ncomp); % Array to store the variances of the depth 
    tempdata = zeros(Niter,N,Ncomp);
    MD = zeros(Niter,Ncomp);  % Initialization MD

    for Nc = 1:Ncomp
        M2=floor(N/2); % mean of the depth prior when initizing (this can be changed)
        S2=N^2/12; % variance of the depth prior when initizing (this can be changed) % m and s2 close to the uniform distribution

        %% Forward estimation
        for t=1:Niter
            Y=data(t,:); % load current frame
            % Main algorithm
            [M2,S2]=online_2D(Y,F_mat,LFM,Fbeta_mat,Falpha_mat,F_sAB,ds,M2,S2,IntFbeta,beta,alpha,div);
            % Store values
            tf(t,Nc)=round(M2);
            stf(t,Nc)=S2;
        end


        %% Backward estimation
        for t=Niter:-1:1
            Y=data(t,:); % load current frame
            % Main algorithm
            [M2,S2]=online_2D(Y,F_mat,LFM,Fbeta_mat,Falpha_mat,F_sAB,ds,M2,S2,IntFbeta,beta,alpha,div);
            % Store values
            tf(t,Nc)=round(M2);
            stf(t,Nc)=S2;
        end

        % remove ridge using the three sigma rule of thumb
        % Computation of the ridge to remove

        for p = 1:Niter
            tempdata(p,:,Nc) = 2*data(p,max(tf(p,Nc)-1,1)).* sqrt(F_mat(:,tf(p,Nc)) ./max(F_mat(:,tf(p,Nc))));
        end

        if ifplot
            figure(2)
            subplot(Ncomp,2,(Nc-1)*2+1)     
            imagesc(transpose(data))
            yticklabels({'200','100','0'})
            yticks([50,150,250])
            title('Current data')
            subplot(Ncomp,2,(Nc-1)*2+2)
            imagesc(transpose(squeeze(tempdata(:,:,Nc))));
            yticklabels({'200','100','0'})
            yticks([50,150,250])
            title(strcat([num2str(Nc),'th estimated ridge']))
        end

        % Update the data without the just estimated rifge
        data = max(data - tempdata(:,:,Nc), eps);

    end


    %% Estimation noise mean
    LBcgk = estimB(transpose(data0),tf,PneiMask);
    LBcgk = LBcgk * N;

    %% Estimation amplitude
    A_hat = estim_amplitude(data0,tf,LBcgk);
    A_hat = sqrt(A_hat); 


    %% Detection for MCS
    if detect == 1
        [MD,~]=detect_MCS(data,Ncomp,tempdata,LBcgk,A_hat,F_mat,tf);
    end

    %% compute mask
    Mask_out = compMask(tf,Pnei,N,0);

    if detect == 1    
        for Nc = 1:Ncomp
            Mask_out(:,:,Nc) = Mask_out(:,:,Nc).* (ones(M,1)*MD(:,Nc)');
        end
    end

else

    [F_mat]=compForacle_STFT(M,L);

    %% Estimation noise mean
    LBcgk = estimB(transpose(data),tf0,20);

    %% Estimation amplitude
    A_hat = estim_amplitude(data,tf0,LBcgk);
    A_hat = sqrt(A_hat); 
    A_hat(tf0==1) = 0;

    % LBcgk = LBcgk * N;

    %% Detection for MCS
    p0 = zeros(Niter,1);      % Initialization p0
    p1=p0;                    % Initialization p1

    %% Matrix precomputation
    w_int = linspace(1e-4,1-1e-4,200);
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
        [P_d,p0(t),p1(t)] = Fdetect(Y,P_d,LBcgk,A_hat(t),cdw,Ftl(:,tf0(t),:));
        P_d = min(max(P_d,1e-3),1-1e-3);
        MD(t) = P_d>0.5;
    end

end