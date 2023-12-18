function [class,K,features,zeros_aux] = classify_spectrogram_zeros_2(zeros_hist, zeros_pos, J, Klist, criterion, plot_figures)
%--------------------------------------------------------------------------
% JMM: This version uses the voronoi cells as neighborhood.
%--------------------------------------------------------------------------
% Classify the zeros of the spectogram of a signal given the 2d histogram
% of zeros, computed using compute_zeros_histogram() function.
%
% Other functions needed:
% - compute_centroids()
%
% Input:
% - zeros_hist: Histogram of zeros computed using the function:
%               'compute_zeros_histogram()'.
% - zeros_pos:  A [N,2] array with the time-frequency coordenates of the
%               zeros of the spectrogram. Where N is the number of zeros.
% - criterion:  A two-elements cell with a combination of and clustering 
%               algorithm ('gmm' or 'knn') and a number of clusters 
%               criterion ('gap' or 'ch'). Examples:
%               -   {'gmm','gap'}
%               -   {'knn','ch'}
% - Klist:      List of integers indicating the number of clusters (K) to
%               test using the indicated criterion.
% - plot_figures: If true, plots the feature space. (Default false).
%
% Output:
% - class:      A [N,1] vector with assigned kind of zeros (1,2 or 3).
% - K:          Number of clusters detected. K=1 means only noise. K=2
%               means a signal is present. K=3 means that zeros of
%               interference between components are present.
% - features:   A [N,2] array with the values of the features computed
%               for each zero.
%
% Example:
%      N = 2^9;
%      x = real(fmlin(N,0.10,0.25)+fmlin(N,0.15,0.3)).*tukeywin(N,0.1);
%      xn = sigmerge(x,randn(size(x)),20);
%      [zeros_hist, zeros_pos, F, w, T, N, S] =...
%                            compute_zeros_histogram(xn, 'estimate');
%      [class,K]  = classify_spectrogram_zeros(zeros_hist, ...
%                                                 zeros_pos, 3*round(T/8));
%      zeros_hist = flipud(zeros_hist);S = flipud(S);
%      zeros_pos(:,1) = N + 1 - zeros_pos(:,1);
%      colores = string({'b';'g';'r'}); symbols = string({'o';'^';'d'});
%      figure(); imagesc(-log(abs(S).^2)); hold on;
%      for i = 1:K
%        plot(zeros_pos(class==i,2),zeros_pos(class==i,1),symbols(i),...
%        'Color',colores(i),'MarkerFaceColor',colores(i),'MarkerSize',2,...
%        'DisplayName', 'Kind '+string(i));
%      end
%      title('Spectrogram and classified zeros'); legend();
%      xticklabels([]); yticklabels([]); xticks([]); yticks([]);
%      xlabel('time'); ylabel('frequency'); colormap pink;
%
% September 2023
% Author: Juan M. Miramont-Taurel <juan.miramont@univ-nantes.fr>
% -------------------------------------------------------------------------


if nargin<4
    plot_figures = false;
end

u=zeros_pos(:,1);
v=zeros_pos(:,2);

[V,C] = voronoin([u,v]);

zeros_aux = [];
% Compute the descriptors for each zero.
for i = 1:size(zeros_pos,1)
    
    % Get coordinates of the VC vertices.
    vxy = round(V(C{i},:));
    
    %     if size(vxy,1)>1
    %         [vxy_aux,~,ic] = unique(vxy,'rows');
    %         ic = ic(diff([ic;ic(1)])>0);
    %         vxy = vxy_aux(ic,:);
    %     end
    
    % Stablish a rectangular domain where the VC lives.
    max_xy = max(vxy); max_x = max_xy(2); max_y = max_xy(1);
    min_xy = min(vxy); min_x = min_xy(2); min_y = min_xy(1);
    
    % Do not consider cells in the border of the TF plane.
    if min_x < 1||min_y < 1 || max_x > size(zeros_hist,2)|| max_y>size(zeros_hist,1)
        %         vxy = fix_this_cell(vxy,zeros_hist);
        %         if any(C{i}==1)
        %             vxy(C{i}==1,:)=[];
        %         end
        continue;
    end
    
    
    %
    %
    %     plot(round(vxy(:,2)),round(vxy(:,1)),'w'); hold on
    
    % If it reached this point, this is a valid zero.
    zeros_aux = [zeros_aux; i];
    local_dist = [];
    
    
    % Extract patch.
    % TODO: FIX THIS TO CONSIDER UNBOUNDED CELLS
    dx = 0; dy = 0; cx = 0; cy = 0; patch_sub = [];
    for qq = max([min_y,1]):min([max_y, size(zeros_hist,1)])
        for pp = max([min_x,1]):min([max_x, size(zeros_hist,2)])
            patch_sub = [patch_sub; qq,pp];
        end
    end
    
    
    patch_ind = sub2ind(size(zeros_hist),patch_sub(:,1),patch_sub(:,2));
    point_in_vc = inpolygon(patch_sub(:,1),patch_sub(:,2),vxy(:,1),vxy(:,2));
    cell_ind = patch_ind(point_in_vc);
    patch = zeros_hist(cell_ind);
    
    dist_ind = cell_ind(patch>0);
    [dist_row,dist_col]= ind2sub(size(zeros_hist),dist_ind);
    
    dist_sub = [dist_row dist_col];
    
    %     local_dist = [];
    %     for j = 1:size(dist_sub,1)
    %         local_dist = [local_dist; ones(zeros_hist(dist_row(j),dist_col(j)),2).*dist_sub(j,:)];
    %     end
    
    try
        ch_vert = convhull(dist_sub(:,1), dist_sub(:,2));
        ch_idx =  dist_sub(ch_vert,:);
    catch
        dist_row = [dist_row; dist_row; dist_row+randi(1)];
        dist_col = [dist_col; dist_col-randi(1); dist_col];
        dist_sub = [dist_row dist_col];
        ch_vert = convhull(dist_sub(:,1), dist_sub(:,2));
        ch_idx =  dist_sub(ch_vert,:);
        warning('Not enough points for CH computation.');
    end
    % Uncomment to check the patches
    %     mask(cell_ind) = 1;
    %     imagesc(mask); hold on;
    %     vxy = [vxy; vxy(1,:)];
    %     plot(round(vy),round(vx),'w');
    %     plot(round(vxy(:,2)),round(vxy(:,1)),'r');
    %     plot(round(ch_idx(:,2)),round(ch_idx(:,1)),'g'); hold on
    
    
    % %  Extract features -------------------------------------------------
    %     dx = mean(local_dist(:,1)); dy = mean(local_dist(:,2));
    %     mass_center = [dx dy];
    
    %     D = pdist(vxy);
    %     max_d = max(D);
    
    %     pgon = polyshape(vxy(:,1),vxy(:,2));
    %     [cx,cy] = centroid(pgon);
    %     zy = zeros_pos(i,2);
    %     zx = zeros_pos(i,1);
    
    %     pgon = polyshape(ch_idx(:,1),ch_idx(:,2));
    %     [cx,cy] = centroid(pgon);
    
    % Area ratio
    vc_area(i,1) = polyarea(vxy(:,1),vxy(:,2));
    ch_area(i,1) = polyarea(ch_idx(:,1),ch_idx(:,2));
    area_ratio(i,1) = ch_area(i,1) / vc_area(i,1);
    
    
    %     dists = sqrt(sum((ch_idx - mass_center).^2,2))/ch_area(i,:);
    %     farp(i,1) = mean(dists)/ch_area;
    %     farp(i,1) = std(dists)/mean(dists);
    %     farp(i,1) = max(dists);
    
    
    %     barydist(i,1) = sqrt((cx-dx)^2 + (cy-dy)^2)/vc_area(i,:);
    %     barydist(i,1) = abs(sqrt((cx-dx)^2 + (cy-dy)^2) - sqrt((zx-cx)^2 + (zy-cy)^2));% / ...
    %                     abs(sqrt((cx-dx)^2 + (cy-dy)^2) + sqrt((zx-dx)^2 + (zy-dy)^2));
    %
    %     plot(cy,cx,'*','Color','c','MarkerFaceColor','b','MarkerSize',6);
    %     plot(dy,dx,'v','Color','m','MarkerFaceColor','m','MarkerSize',6);
    
    
    
    % Skweness
    %         skw(i,1) = mardia_skw(local_dist, [0,0]); %zeros_pos(i,:));
    %     skw(i,1) = sqrt(sum((median(local_dist)-mean(local_dist)).^2));
    
    % Local Density measures:
    %     patch_aux = exp(patch(:)/512)-1;
    %     norm_1(i,1) = sum(patch_aux)/polyarea(vxy(:,1),vxy(:,2));
    %     norm_0(i,1) = sum(patch(:)==0)/length(patch);
    %             max_patch(i,1) = exp(max(patch)/512)-1;
    max_patch(i,1) = max(patch)/J;
    %     mean_patch(i,1) = median(patch(patch>0));
    %                 cvar(i,1) = std(patch(patch>0))/mean(patch(patch>0));
    %                 cvar(i,1) = std(patch(patch>0))/mean(patch(patch>0));
    %     cvar(i,1) = max(patch)/ch_area(i,1);
    
    % Local Concentration measures:
%     patch_blurred = zeros_hist_blurred(cell_ind);
%     S = sum(patch_blurred(:));  % Normalize the sum within the neighboor.
    %     entropy(i,1) = -sum(log2(((patch_blurred(:)+eps)/S)).*(patch_blurred(:)/S)); % Shannon
    %           entropy_min(i,1) = -log2(max(patch(:)/S)); % MinEntropy
    %                 entropy(i,1) = -log(sum((patch(:)/S+eps).^2)); % Collision
    
end

features = [area_ratio, max_patch];

% keep only those zeros with a well defined VC.
features = features(zeros_aux,:);
zeros_aux = zeros_pos(zeros_aux,:);
% features = normalize(features);

% -------------------------------------------------------------------------
% Determine the number and form the clusters
% -------------------------------------------------------------------------

[K,class] = find_number_of_clusters(features, criterion, Klist);


[~,C] = compute_centroids(features,class);

% Labeling
if K==1
    class(:) = 2;
end

if K==3
    
    % Sorting clusters by ascending norm:
    [~,sorted_cluster_norm] = sort(C(:,1),'ascend');
    cluster_lowest_norm = sorted_cluster_norm(1);
    cluster_mid_norm = sorted_cluster_norm(2);
    cluster_highest_norm = sorted_cluster_norm(3);
    
    % Lowest area -> Class SS zeros.
    class(class==cluster_lowest_norm) = 10;
    
    % Highest area -> Class NN zeros
    class(class==cluster_highest_norm) = 20;
    
    % Mid area -> Class SN zeros
    class(class==cluster_mid_norm) = 30;
    
    class = class/10;
    
    C = C(sorted_cluster_norm,:);
    
end

if K==2
    
    % Sorting clusters by ascending norm:
    [~,sorted_cluster_norm] = sort(C(:,1),'ascend');
    cluster_lowest_norm = sorted_cluster_norm(1);
    cluster_highest_norm = sorted_cluster_norm(2);
    
    % Lowest area -> Class SN zeros.
    class(class==cluster_lowest_norm) = 30;
    
    % Highest area -> Class NN zeros
    class(class==cluster_highest_norm) = 20;
    
    class = class/10;
    
    C = C(sorted_cluster_norm,:);
    
end


if plot_figures
    figure()
    for i = unique(class).'
        plot(features(class==i,1),...
            features(class==i,2),'o'); hold on;
    end
    
    
    plot(C(:,1),C(:,2),'o','MarkerFaceColor','c',...
        'MarkerEdgeColor','m',...
        'MarkerSize',4,...
        'DisplayName','Centroids');
    
    
    grid on;
    xlabel('$\Vert G_{z} \Vert_{1}$','Interpreter','latex');
    ylabel('$H_{\infty}(G_{z})$','Interpreter','latex');
    % xticklabels([]); yticklabels([])
    % xticks([]); yticks([])
    legend('boxoff');
    legend('Location','southwest');
    
end

end


