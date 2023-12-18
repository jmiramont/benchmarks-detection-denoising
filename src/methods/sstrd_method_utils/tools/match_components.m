function [ I, s ] = match_components(X, Y, criterion)
% function [ I, s ] = match_components(X, Y, criterion)
%
% Match each component corresponding to a row vector of X with a component
% of Y
%
% Input:
% X: Reference signals (Matrix)
% Y: Reconstructed signals
% criterion: pointer to a function to maximize (defaut:@SNR)
%
% Output:
% I: index of each component (e.g. Y(I(1), :) corresponds to X(1, :)) 
% s: maximal score of each matched component  s(i) = @criterion(X(I(i),:), )

if ~exist('criterion', 'var')
 criterion = @SNR;
end

if norm(size(X) - size(Y)) > 0
 error('X and Y should have the same size');
end

nc = size(X,1);  % number of components

I  = zeros(1, nc);
s  = -ones(1, nc) * inf;

for i = 1:nc
 for j = 1:nc
    tmp = criterion(X(i,:), Y(j,:));
    if tmp > s(i)
      s(i) = tmp;
      I(i)  = j;
    end
 end
end


dup     = get_duplicate( I ); %% indices of duplicate values
I(dup)  = 0;

d = setdiff(1:nc, I);  %% values of missing components
z = find(I == 0);      %% index of non affected values or duplicate values

if ~isempty(z)|| ~isempty(d)
 warning('duplicate or non affected components');
 
 %% distribute non affected values on zeros
 if length(d) == length(z)
  for i = 1:length(d)
   I(z(i)) = d(i);
   s(z(i)) = criterion(X(z(i),:), Y(d(i),:));
  end
 end
end






