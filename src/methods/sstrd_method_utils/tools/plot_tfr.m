function [ ] = plot_tfr(tfr, t, f )

 if ~exist('t','var') || ~exist('f','var')
    set(gca,'YTick',[]);set(gca,'XTick',[])
    imagesc(tfr)
 else
    imagesc(t, f, tfr);
 end
 
set(gca,'YDir','normal')
colormap gray;
cmap = flipud(colormap);
colormap(cmap);

 xlabel('time')
 ylabel('frequency')
 