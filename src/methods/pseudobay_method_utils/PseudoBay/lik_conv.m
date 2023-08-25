function [out] = lik_conv(y,Ftl)

Lc = size(Ftl,3);
M = length(y);

% out = y*log(reshape(M.*Ftl,size(Ftl,1),Lc) + 1);

out = y*log(reshape(Ftl,size(Ftl,1),Lc) + (1/M));
% out = (y*reshape(Ftl,size(Ftl,1),Lc) + 1);
% out = (y*reshape(Ftl,size(Ftl,1),Lc) + (1/M));



out = exp(out);

out = out';



end