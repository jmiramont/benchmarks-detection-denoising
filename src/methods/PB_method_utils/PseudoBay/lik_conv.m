function [out] = lik_conv(y,Ftl)

Lc = size(Ftl,3);
M = length(y);

out = y*log(reshape(Ftl,size(Ftl,1),Lc) + 1);
out = exp(out);

out = out';

end