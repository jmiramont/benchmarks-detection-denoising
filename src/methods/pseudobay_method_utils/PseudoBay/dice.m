function [ val ] = dice(X1, X2)
% function [mask] = dice(X1, X2)
%
% Compute the Sorensen-dice index similarity measure between binary matrices X1 and X2
%
% INPUT:
% X1,X2 : matrices of the same size to compare
%
%
% OUTPUT:
% mask: binary matrix of same dimension of tfr with values in {0,1}
%
%
% Author: D.Fourer (dominique.fourer@univ-evry.fr)
% Date: 15-feb-2021


[M1,N1] = size(X1);
[M2,N2] = size(X2);

if M1~=M2 || N1~=N2
  error('X1 and X2 should have the same dimension');
end

if ~check_vals(X1) || ~check_vals(X2)
  error('X1 and X2 should be binary (only values in {0,1})');
end

val = 2 * nb_elem(X1 .* X2) / (nb_elem(X1==1)+nb_elem(X2==1));
    
end


function [val] = check_vals(X)
    val = ((nb_elem(X==0) + nb_elem(X==1)) == numel(X));
end

function [val] = nb_elem(X)
  val = length(find(X));  
end


