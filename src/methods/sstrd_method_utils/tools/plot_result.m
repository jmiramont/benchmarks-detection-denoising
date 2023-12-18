function [I, s] = plot_result( X, Y, method_name, lims )
%function [  ] = plot_result( X, Y )
% Plot the reference signals and their reconstructions
%
% Input:
% X: Reference signals (Matrix)
% Y: Reconstructed signals
%
% Output:
% I: index of each component (e.g. Y(I(1), :) corresponds to X(1, :)) 
% s: maximal SNR of each matched component  s(i) = SNR(X(I(i),:), )


if ~exist('method_name', 'var')
 method_name = 'Method'; 
end

if ~exist('lims', 'var')
 lims = [];
end

nc = size(X,1);

if size(X,1) ~= size(Y,1);
 Y = Y';
end

[ I, s ] = match_components(X, Y);
for i = 1:nc
  subplot(nc, 1, i);
  plot(X(i, :), 'g-.');
  hold on
  plot(Y(I(i),:), 'k-');
  if ~isempty(lims)
   ylim(lims)   
  end
  xlabel('time index')
  ylabel('amplitude')
  title(sprintf('%s: component %d, reconstruction RQF=%.2f dB',method_name, i, s(i)));
  legend('reference', 'reconstruction')
  hold off
end

end

