function [withiness,centroids] = compute_centroids(descriptors,assigned_cluster)
% Compute the centroid and withiness of the clusters from data given in
% descriptors, using the labels given in assigned clusters.

labels = unique(assigned_cluster);
for i = 1:length(labels)
    X = descriptors(assigned_cluster == labels(i),:);
    centroids(i,:) = mean(X,1); 
    p = 2;
    dist_fun = @(x)norm(x,p);
    d = cellfun(dist_fun,num2cell(X-centroids(i,:),2));
    C(i) = mean(d);
end
try
withiness = sum(C);
catch
    disp('Hola');
end

