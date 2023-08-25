function [ IF ] = mask2if( mask, threshold )

if ~exist('threshold', 'var')
  threshold = 0.5;    
end

[M,N] = size(mask);

IF = zeros(1,N);

for n = 1:N
 I = find (mask(:,n) > threshold);   
 IF(n) = mean(I);  
end


end

