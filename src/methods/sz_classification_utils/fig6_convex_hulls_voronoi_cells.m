%% Example movement of zeros and histograms
clear all; close all;
% Generate a pair of parallel chirps close to each other.
rng(0)
N = 2^8;
Nchirp = N;
tmin = round((N-Nchirp)/2);
tmax = tmin + Nchirp;
x = zeros(N,1);
instf1 = 0.1+0.2*(0:Nchirp-1)/Nchirp;
instf2 = 0.15+0.2*(0:Nchirp-1)/Nchirp;
x(tmin+1:tmax) = (cos(2*pi*cumsum(instf1)) + cos(2*pi*cumsum(instf2))).*tukeywin(Nchirp,0.25).';

% Parameters for the STFT.
Nfft = 2*N;
fmax = 0.5; % Max. norm. frequency to compute the STFT.
[w,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.

dT = 3*round(T/8);

% Noise realization:
SNRin = 30;
original_noise = randn(size(x));
[xnoise,std_noise] = sigmerge(x,original_noise,SNRin);
[F,~,~] = tfrstft(xnoise,1:N,Nfft,w,0);
F = F(1:floor(Nfft*fmax),:);
F = flipud(F);
S = abs(F).^2;

% Find original zeros and triangulation
[ceros,Qz] = find_spectrogram_zeros(S);
% TRI = delaunay(ceros);

% Voronoi Cells
[V,C] = voronoin(ceros);
[vx,vy] = voronoi(ceros(:,1),ceros(:,2));

figure()
imagesc((abs(F))); hold on;
plot(ceros(:,2),ceros(:,1),'wo','MarkerFaceColor','w','MarkerSize',1.5);
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
% print_figure('example_ascete_zeros_movement_spectrogram.pdf',5,5)

%%
% noise_alg = randn(N,1);
% noise_alg = noise_alg/std(noise_alg)*std_noise;
% xnoise_alg = x+noise_alg;
% %     xnoise_alg = sigmerge(xnoise,noise_alg,SNRalg);
% [S_alg,~,~] = tfrsp(xnoise_alg,1:N,Nfft,w,0);
% S_alg = S_alg(1:floor(Nfft*fmax),:);
% S_alg = flipud(S_alg);
% [new_zeros, Qz] = find_zeros_stft(S_alg);
% aux_plane = aux_plane + Qz;


%% Histogram with more noise realizations each time
disp('Computing histogram...');
lims = 0;
aux_plane = zeros(size(S));
J = 2048; % Number of noise realizations.
% tic()
parfor j  = 1:J
    noise_alg = randn(N,1);
    noise_alg = noise_alg/std(noise_alg)*std_noise;
    xnoise_alg = xnoise+noise_alg;
    %     xnoise_alg = sigmerge(xnoise,noise_alg,SNRalg);
    [S_alg,~,~] = tfrsp(xnoise_alg,1:N,Nfft,w,0);
    S_alg = S_alg(1:floor(Nfft*fmax),:);
    S_alg = flipud(S_alg);
    [~, Qz] = find_spectrogram_zeros(S_alg);
    aux_plane = aux_plane + Qz;
end
% toc()
hist2d = aux_plane(lims+1:end-lims,lims+1:end-lims);
zeros_hist = hist2d;

figure(); imagesc(log(zeros_hist)); colormap gray; hold on;
plot(round(vy),round(vx),'w');
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
% print_figure('example_ascete_zeros_movement_hist_1024.pdf',5,5)

%%

figure(); %subplot(4,3,1:9);
imagesc(log(zeros_hist)); hold on; colormap gray;
plot(ceros(:,2),ceros(:,1),'r.','MarkerSize',4); hold on;
plot(ceil(vy),ceil(vx),'w');
xticklabels([]); yticklabels([])
xticks([]); yticks([])
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
% Plot the dots marking the zeros.
plot(ceros([40,46,101],2),ceros([40,46,101],1),'o','Color','r','MarkerFaceColor','r',...
    'MarkerSize',2.0); hold on;
% Plot the corresponding VCs
% viscircles(fliplr(ceros([40,46,85],:)),dT*ones(3,1),'LineWidth', 0.25, 'EnhanceVisibility',false);

for i = [40, 46, 101]
    % Get coordinates of the VC vertices.
    vxy = ceil(V(C{i},:));
    vxy = [vxy; vxy(1,:)];
    plot(round(vxy(:,2)),round(vxy(:,1)),'r','LineWidth',1.25);

end

text(ceros([40,46,101],2)+1,ceros([40,46,101],1),{'NN','SS','SN'},...
    "FontSize",5.0,...
    'Color','r',...
    'FontWeight','bold',...
    'HorizontalAlignment','right',...
    'VerticalAlignment','bottom');

% print_figure('example_ascete_zeros_neighborhood_VC.pdf',7.2,7.2,'RemoveMargin',true)

%%
figure()
imagesc(ones(size(zeros_hist))); hold on; colormap bone; clim([0,1]);
plot(ceil(vy),ceil(vx),'k');
xticklabels([]); yticklabels([])
xticks([]); yticks([]);

% xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
plot(ceros(40,2),ceros(40,1),'o','Color','r','MarkerFaceColor','r',...
    'MarkerSize',0.5); hold on;

% Get coordinates of the VC vertices.
vxy = ceil(V(C{40},:));
vxy = [vxy; vxy(1,:)];
plot(round(vxy(:,2)),round(vxy(:,1)),'r','LineWidth',2.0);

% Stablish a rectangular where the VC lives.
max_xy = max(vxy); max_x = max_xy(1); max_y = max_xy(2);
min_xy = min(vxy); min_x = min_xy(1); min_y = min_xy(2);

text(ceros(40,2)+18,ceros(40,1)-8,{'NN'},'Color','k',...
    'FontWeight','bold',...
    'HorizontalAlignment','right',...
    'VerticalAlignment','bottom');

xlim([min_y-5, max_y+5]);
ylim([min_x-5, max_x+5]);



patch_sub = [];
for qq = min_x:max_x
    for pp = min_y:max_y
        patch_sub = [patch_sub; [qq,pp] ];
    end
end

patch_ind = sub2ind(size(zeros_hist),patch_sub(:,1),patch_sub(:,2));
point_in_vc = inpolygon(patch_sub(:,1),patch_sub(:,2),vxy(:,1),vxy(:,2));
cell_ind = patch_ind(point_in_vc);
patch = zeros_hist(cell_ind);

dist_ind = cell_ind(patch>0);
[dist_row,dist_col]= ind2sub(size(zeros_hist),dist_ind);

dist_sub = [dist_row dist_col];

% plot(round(dist_sub(:,2)),round(dist_sub(:,1)),'k.');

ch_vert = convhull(dist_sub(:,1), dist_sub(:,2));
ch_idx =  dist_sub(ch_vert,:);
%
fill(round(ch_idx(:,2)),round(ch_idx(:,1)),'red','FaceAlpha',0.3,'LineStyle','none');
plot(round(vxy(:,2)),round(vxy(:,1)),'k','LineWidth',2.0);



% print_figure('convex_hull_zoom_NN_kind.pdf',2,2,'RemoveMargin',true)

%%
figure()
imagesc(ones(size(zeros_hist))); hold on; colormap bone; clim([0,1]);
plot(ceil(vy),ceil(vx),'k');
xticklabels([]); yticklabels([])
xticks([]); yticks([]);

% xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
plot(ceros(46,2),ceros(46,1),'o','Color','r','MarkerFaceColor','r',...
    'MarkerSize',0.5); hold on;

% Get coordinates of the VC vertices.
vxy = ceil(V(C{46},:));
vxy = [vxy; vxy(1,:)];
plot(round(vxy(:,2)),round(vxy(:,1)),'k','LineWidth',2.0);

% Stablish a rectangular where the VC lives.
max_xy = max(vxy); max_x = max_xy(1); max_y = max_xy(2);
min_xy = min(vxy); min_x = min_xy(1); min_y = min_xy(2);

text(ceros(46,2)+18,ceros(46,1)-8,{'SS'},'Color','k',...
    'FontWeight','bold',...
    'HorizontalAlignment','right',...
    'VerticalAlignment','bottom');

xlim([min_y-2, max_y+2]);
ylim([min_x-2, max_x+2]);

patch_sub = [];
for qq = min_x:max_x
    for pp = min_y:max_y
        patch_sub = [patch_sub; [qq,pp] ];
    end
end

patch_ind = sub2ind(size(zeros_hist),patch_sub(:,1),patch_sub(:,2));
point_in_vc = inpolygon(patch_sub(:,1),patch_sub(:,2),vxy(:,1),vxy(:,2));
cell_ind = patch_ind(point_in_vc);
patch = zeros_hist(cell_ind);

dist_ind = cell_ind(patch>0);
[dist_row,dist_col]= ind2sub(size(zeros_hist),dist_ind);

dist_sub = [dist_row dist_col; dist_row dist_col+1; dist_row dist_col-1;
            dist_row+1 dist_col; dist_row-1 dist_col];

% plot(round(dist_sub(:,2)),round(dist_sub(:,1)),'k.');

ch_vert = convhull(dist_sub(:,1), dist_sub(:,2));
ch_idx =  dist_sub(ch_vert,:);
%
fill(round(ch_idx(:,2)),round(ch_idx(:,1)),'red','FaceAlpha',0.3,'LineStyle','none');
plot(round(vxy(:,2)),round(vxy(:,1)),'k','LineWidth',2.0);


% print_figure('convex_hull_zoom_SS_kind.pdf',2,2,'RemoveMargin',true)
%%
figure()
imagesc(ones(size(zeros_hist))); hold on; colormap bone; clim([0,1]);
plot(ceil(vy),ceil(vx),'k');
xticklabels([]); yticklabels([])
xticks([]); yticks([]);

% xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
plot(ceros(101,2),ceros(101,1),'o','Color','r','MarkerFaceColor','r',...
    'MarkerSize',0.5); hold on;

% Get coordinates of the VC vertices.
vxy = ceil(V(C{101},:));
vxy = [vxy; vxy(1,:)];


% Stablish a rectangular where the VC lives.
max_xy = max(vxy); max_x = max_xy(1); max_y = max_xy(2);
min_xy = min(vxy); min_x = min_xy(1); min_y = min_xy(2);

text(ceros(101,2)+15,ceros(101,1),{'SN'},'Color','k',...
    'FontWeight','bold',...
    'HorizontalAlignment','right',...
    'VerticalAlignment','bottom');

xlim([min_y-5, max_y+5]);
ylim([min_x-5, max_x+5]);

patch_sub = [];
for qq = min_x:max_x
    for pp = min_y:max_y
        patch_sub = [patch_sub; [qq,pp] ];
    end
end

patch_ind = sub2ind(size(zeros_hist),patch_sub(:,1),patch_sub(:,2));
point_in_vc = inpolygon(patch_sub(:,1),patch_sub(:,2),vxy(:,1),vxy(:,2));
cell_ind = patch_ind(point_in_vc);
patch = zeros_hist(cell_ind);

dist_ind = cell_ind(patch>0);
[dist_row,dist_col]= ind2sub(size(zeros_hist),dist_ind);

dist_sub = [dist_row dist_col];

% plot(round(dist_sub(:,2)),round(dist_sub(:,1)),'k.');

ch_vert = convhull(dist_sub(:,1), dist_sub(:,2));
ch_idx =  dist_sub(ch_vert,:);
%
fill(round(ch_idx(:,2)),round(ch_idx(:,1)),'red','FaceAlpha',0.3,'LineStyle','none');
plot(round(vxy(:,2)),round(vxy(:,1)),'k','LineWidth',2.0);

% print_figure('convex_hull_zoom_SN_kind.pdf',2,2,'RemoveMargin',true)

