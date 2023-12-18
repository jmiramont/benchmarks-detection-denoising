function [ I ] = get_duplicate( X )
%function [ I ] = get_duplicate( X )
%
% return the indices of duplicate values in vector X
%
% This function is faster than
% [~,tmp] = unique(I);
% dup     = setdiff(1:length(I), tmp);

I = [];

for i = 1:(length(X)-1)
    
 if ~isempty(find(I == i, 1)) %% already found
   continue;  
 end
 
 dup = find( X == X(i));
 if length(dup) > 1
  I = [I dup(2:end)];
 end
end

end

