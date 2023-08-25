function [Mask_out,tf,AltS,A_hat,LBcgk,MD] = pseudoBay_conv(tfr,Ncomp,M,L,div,beta,alpha,ds,Pnei,ifplot,detect,sig_conv,PneiMask)
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
% Mask_out      : Mask of five bins per ridge. Modify code below to change
%                 the neighborhood


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
 Pnei=4;% default KL
end


data = transpose(abs(tfr(1:round(M/2),:))).^2+ eps; % Absolute value of the top half TFR


[Niter,N] = size(data); % Extract dimenssions
data0=data;

[F_mat, Fbeta_mat, IntFbeta, Falpha_mat, varF, F_sAB]=compFbeta_STFT(beta,alpha,M,L,N);
LFM = log(F_mat+eps);


% FF = (F_mat(:,150)); FF=FF./max(FF);
% dd = data(150,:); dd=dd./max(dd);
% 
% [~,mt]=max(FF);
% [~,md]=max(dd);
% 
% figure(1)
% hold on
% plot(dd,'b')
% plot(circshift(FF,md-mt),'g')
% legend('data','irf')

%% Initialization
tf=zeros(Niter,Ncomp); %Array to store the means of the depth
stf=zeros(Niter,Ncomp); % Array to store the variances of the depth 
tempdata = zeros(Niter,N,Ncomp);
AltS = zeros(N,Niter);
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
    
%     tempdata(:,:,Nc) = zeros(size(data));
    for p = 1:Niter
        tempdata(p,:,Nc) = data(p,max(tf(p,Nc)-1,1)).* F_mat(:,tf(p,Nc)) ./max(F_mat(:,tf(p,Nc)));
%         tempdata(p,:,Nc) = 2*data(p,max(tf(p,Nc)-1,1)).* sqrt(F_mat(:,tf(p,Nc)) ./max(F_mat(:,tf(p,Nc))));
%         tempdata(p,max(tf(p,Nc)-1-Pnei*round(varF),1):min(tf(p,Nc)-1+Pnei*round(varF),N),Nc) = 1;
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
%     data(tempdata(:,:,Nc)==1) = 0;
    data = max(data - tempdata(:,:,Nc), 0);

    
end


%% Estimation noise mean
LBcgk = estimB(transpose(data0),tf, PneiMask);


%% Estimation amplitude
A_hat = estim_amplitude(data0,tf,LBcgk);
A_hat = sqrt(A_hat); 




% Pnei = 10;
Mask = compMask(tf,Pnei,N,0);



Mask_out = Mask;
if detect == 1    
    for Nc = 1:Ncomp
        vec = find(MD(:,Nc));
        for tt=1:length(vec)
            t = vec(tt);
            AltS(:,t)= AltS(:,t) + A_hat(t,Nc).* F_mat(:,tf(t,Nc)) ./max(F_mat(:,tf(t,Nc)));
        end
        Mask_out(:,:,Nc) = Mask_out(:,:,Nc).* (ones(M,1)*MD(:,Nc)');
    end
else
    for Nc = 1:Ncomp
        for t=1:Niter
            AltS(:,t)= AltS(:,t) + A_hat(t,Nc).* F_mat(:,tf(t,Nc)) ./max(F_mat(:,tf(t,Nc)));
            temnormpdf = normpdf(1:M,max(tf(t,Nc)-1,1),sig_conv)';
            Mask_out(:,t,Nc) = Mask(:,t,Nc).* (temnormpdf./max(temnormpdf));
        end
    end
end

Mask_out = Mask_out(1:M/2,:,:);
Mask_out = [Mask_out;Mask_out(end:-1:1,:,:)];



