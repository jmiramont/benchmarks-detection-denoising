function [prob,p0,p1] = Fdetect2(y,pi0,b,A,cdw,Ftl,M)
%
% Detection step - returns the probability of a frequency component associated
%                  with the signal to be present in the observation
%
% INPUT:
% y           : data
% pi0         : prior probability of a presence of abscence of a target
% b           : Noise expectation
%
% OUTPUT:
% prob        : Probability of target presence
%

pi0 = 0.5;


if sum(y)==0
    prob = 0.5;
else
    %% priors
    lamba_a =  1/(A+eps);
    lamba_b = 1/(b+eps);
        
    %% densities
    p0 = (1-pi0)/(lamba_b*M);
    p1 = pi0*lamba_a*lamba_b .* sum(( lik_conv(y,Ftl) ./ (((lamba_a*cdw+lamba_b).^2)+eps)));
    
    %% find map
    det = p0/p1;
    prob = 1/(1+det);

end
end