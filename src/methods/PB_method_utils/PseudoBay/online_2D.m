function [mcav_ext,s2cav_ext]=online_2D(Y,F,...
    LFM,Fbeta,Falpha,F_sAB,ds,mcav_ext,s2cav_ext,IntFbeta,beta,alpha,div)
%
% Online approach. Pseudo-Bayesian method for estimating the current (for 
% the current time considered) position of the ridge. The pseudo-likelihood
% term is the chosen divergence and the temporal prior a Gaussian random
% walk centerer around the previous estimated ridge position.
%
% INPUT:
% Y              : data
% F              : data distribution
% LFM            : Precomputed window for KL divergence
% Fbeta          : Precomputed window for beta divergence
% Falpha         : Precomputed window for Renyi divergence
% ds             : variance of the Gaussian random walk
% mcav_ext       : mean of the prior
% s2cav_ext      : standard deviation of the prior
% IntFbeta       : Precomputed sum for beta divergence
% beta           : Divergence parameter
% alpha          : Divergence parameter
% div            : Divergence choice: 1=KL   |   2=beta   |   3=Renyi
% MD             :
% P_d            :
% b              :
%
%
% OUTPUT:
% mcav_ext       : estimated mean (position of the ridge by MMSE)
% s2cav_ext      : estimated variance of the estimate
% A              : estimated reflectivity (if detection)
% b              : estimated background (if detection)
% MD             : detection bolean
% P_d            : 

T = length(Y);
% Y = Y./sum(Y);
%%%%%%%%%%%%%%%%%%%%%%%

%% Compute log-divergence (pseudo-likelihood)
if div == 1          % KL divergence
    U = (Y*LFM); 
elseif div == 2      % beta divergence
    U=(Y*Fbeta); 
    U = ((1+beta)/beta)*U - IntFbeta;
elseif div == 3      % Renyi divergence
    U = -log((Y.^alpha)*(Falpha))./(alpha-1);
elseif div == 4      % sAB divergence
    U =  -(F_sAB)./(alpha*(alpha+beta)) + ((Y.^alpha)*Fbeta)./(alpha*beta);% - (sum(Y.^(alpha+beta)))./(beta*(alpha+beta));
else                 % If none chosen, select KL divergence
    U = (Y*LFM); 
end

%% Compute prior
SS=s2cav_ext+ds;
F1=normpdf(1:T,mcav_ext,sqrt(SS));
%% Compute posterior (combine pseudo-likelihood and prior)
U=U+log(F1+eps);
U=U-max(U,[],2)*ones(1,T);
P=exp(U);
P=P./sum(P);




%% Compute the mean and variance of the posterior
mcav_ext=P*(1:T)';
s2cav_ext=P*((1:T)'.^2)-mcav_ext.^2;

