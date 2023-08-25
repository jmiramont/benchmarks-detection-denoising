function [ ] = plot_comp( S, vec_nc, tfr,lims )
%PLOT_COMP Summary of this function goes here
%   Detailed explanation goes here



if size(S,1) > size(S,2)
 S = S.';
end


 if ~exist('vec_nc', 'var')
  vec_nc = zeros(1, size(S,2));
  for i = 1:size(S, 1)
   I = find(abs(S(i, :)) > eps);
   vec_nc(I) = vec_nc(I) + 1;
  end
 end
 
 if ~exist('tfr', 'var')
   tfr = 0;
 end
 
 if ~exist('lims', 'var')
   lims = [];
 end
 
if ~isempty(vec_nc)
 d = 1;
else
 d = 0;
end

figure(998)
for i = 1:size(S,1)
  subplot(size(S, 1)+d,1,i)
  plot(S(i, :), 'k-')
  if ~isempty(lims)
   ylim(lims)   
  end
  xlabel('time index')
  title(sprintf('Component %d', i))
end

if d == 1
 subplot(size(S, 1)+d,1,size(S, 1)+1)
 plot(vec_nc, 'k-')
 xlabel('time index')
 %ylim([0 max(vec_nc)+0.5])
 title('number of components')
end

s = sum(S);
s = s(:);
if tfr == 1
 figure(999)
 Nh = 81; %127;% short-time window length
 Nf = 256;% # of frequency bins
 w = tftb_window(Nh,'Kaiser');
 [sp,rs] = tfrrsp(s,1:length(s),Nf,w);
 imagesc(flipud(rs(1:128,:).^0.2))
 set(gca,'YTick',[]);set(gca,'XTick',[])
 xlabel('time')
 ylabel('frequency')
 title('reassigned spectrogram of the signal')
end

