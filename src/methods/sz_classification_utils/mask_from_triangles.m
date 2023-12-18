function mask = mask_from_triangles(S,TRI,zeros_pos)
% Generate a 0/1 mask to filter the TF signal domain from the short-time
% Fourier transform. The mask is computed using a set of Delaunay triangles
% computed on the zeros of the spectrogram.
%
% Input:
% - S:          Signal spectrogram.
% - TRI:        A [K,3] matrix, where K is the number of triangles,
%               defined by three zeros of the spectrogram.
% - zeros_pos:  A [N,2] matrix, where N is the number of zeros of the
%               spectrogram.
%
% Ouput:
% - mask:       A 0/1 extraction mask to filter the short-time Fourier
%               transform.
%
% September 2022
% Author: Juan M. Miramont <juan.miramont@univ-nantes.fr>
% -------------------------------------------------------------------------

mask = zeros(size(S));
for i=1:size(TRI,1)
    mask = mask + points_in_triangle(S,TRI(i,:),zeros_pos);
end
mask(mask>0)=1;
if sum(mask(:)) == 0
    mask = ones(size(mask));
end

end

function mask = points_in_triangle(S,TRI,tri_vertex)
% A filling algorithm to determine the points inside a given set of
% triangles.
% Disclaimer: Not the most efficient way to do this, but you don't need
% other Matlab toolboxes.
%--------------------------------------------------------------------------


mask=zeros(size(S));

% Coordenadas de los vertices:
vertTRI = tri_vertex(TRI,:);

AT=(vertTRI(1,1)*(vertTRI(2,2)-vertTRI(3,2))+...
    vertTRI(2,1)*(vertTRI(3,2)-vertTRI(1,2))+...
    vertTRI(3,1)*(vertTRI(1,2)-vertTRI(2,2)));

minX=min(vertTRI(:,1));
maxX=max(vertTRI(:,1));
minY=min(vertTRI(:,2));
maxY=max(vertTRI(:,2));

submasc=mask(minX:maxX,minY:maxY);

for i=1:size(submasc,1)
    for j=1:size(submasc,2)
        punto=[minX+i,minY+j];
        
        A1=(vertTRI(1,1)*(vertTRI(2,2)-punto(2))+...
            vertTRI(2,1)*(punto(2)-vertTRI(1,2))+...
            punto(1)*(vertTRI(1,2)-vertTRI(2,2)));
        
        A2=(vertTRI(1,1)*(punto(2)-vertTRI(3,2))+...
            punto(1)*(vertTRI(3,2)-vertTRI(1,2))+...
            vertTRI(3,1)*(vertTRI(1,2)-punto(2)));
        
        A3=(punto(1)*(vertTRI(2,2)-vertTRI(3,2))+...
            vertTRI(2,1)*(vertTRI(3,2)-punto(2))+...
            vertTRI(3,1)*(punto(2)-vertTRI(2,2)));
        
        
        if sum(abs([A1 A2 A3]))==AT
            submasc(i,j)=1;
        end
        
        
    end
end
mask(minX:maxX,minY:maxY)=submasc;
end