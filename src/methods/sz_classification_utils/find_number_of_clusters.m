function [K,class] = find_number_of_clusters(features,criterion,Klist)

if nargin<3 || isempty(Klist)
    Klist = [2,3]; % Number of noise realizations.
end

% Select a clustering method ----------------------------------------------
if criterion(1) == "kmeans"
    distance = 'cityblock';
    cluster_fun = @(DATA,K) kmeans(DATA,K,...
                                  'Replicates',5,...
                                  'Distance',distance);
end

if criterion(1) == "gmm"
    distance = 'sqEuclidean';
    cluster_fun = @(DATA,K) cluster(fitgmdist(DATA,K,'Replicates',5,...
        'SharedCovariance', false),...
        DATA);
end


% Evaluate the number of clusters -----------------------------------------

if criterion(2) == "gap"
    eva = evalclusters(features, cluster_fun, 'gap', 'KList', Klist,...
                                    'B', 100, 'Distance', distance);
end

if criterion(2) == "ch"
    eva = evalclusters(features,cluster_fun,'CalinskiHarabasz','KList',Klist);
end


K = eva.OptimalK;

% Dealing with ill-conditioned variance matrix ----------------------------
if isnan(K)
    nanflag = true;
    disp('Found NaN. Increase J.');

    if criterion(1) == "gmm"
        cluster_fun = @(DATA,K) cluster(fitgmdist(DATA,K,'Replicates',5,...
            'SharedCovariance', true),...
            DATA);
    end

    if criterion(2) == "gap"
        eva = evalclusters(features, cluster_fun, 'gap', 'KList', Klist,...
            'B', 100, 'Distance', distance);
    end

    if criterion(2) == "ch"
        eva = evalclusters(features,cluster_fun,'CalinskiHarabasz',...
                           'KList',Klist);
    end
    K = eva.OptimalK;
end
%--------------------------------------------------------------------------

% eva = evalclusters(features,cluster_fun,'CalinskiHarabasz','KList',[2,3]);


% cluster_fun = @(DATA,K) clusterdata(DATA,'Distance','cityblock','Maxclust', K);



%                                           'RegularizationValue',0.01),...
% cluster_fun = @(DATA,K) kmeans(Hceros,K,'Replicates',101,'Distance','sqeuclidean');
% cluster_fun = @(DATA,K) kmedoids(Hceros,K,'Replicates',31,'Distance','sqeuclidean');
% eva = evalclusters(features,cluster_fun,'CalinskiHarabasz','KList',[2,3]);
% eva = evalclusters(Hceros,cluster_fun,'DaviesBouldin','KList',[2,3]);
% eva = evalclusters(features,"linkage",'gap','KList',[2,3], 'B', 101); %,'ReferenceDistribution','uniform');
% eva = evalclusters(features,'gmdistribution','gap','KList',[1,2,3],'B', 101);
% save('eva.mat','eva');


% [~,K] = max(eva.CriterionValues);


class = eva.OptimalY;

