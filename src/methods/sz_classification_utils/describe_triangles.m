function [lados,maxlad,TRI_LADOS] = describe_triangles(TRI,vertices,Nfft,T)

Ly = Nfft/T ;
Lx = T ;

% Normalize the coordinates of the vertices.
vertices(:,1) = (vertices(:,1)-1)/Ly;
vertices(:,2) = (vertices(:,2)-1)/Lx;

lados=zeros(size(TRI,1)*3,1);
TRI_LADOS=zeros(size(TRI));
maxlad=zeros(size(TRI,1),1);

k=0;
for i=1:size(TRI,1)
    k=k+1;
    lados(k)=sum((vertices(TRI(i,1),:)-vertices(TRI(i,2),:)).^2);
    TRI_LADOS(i,1)=lados(k);
    k=k+1;
    lados(k)=sum((vertices(TRI(i,1),:)-vertices(TRI(i,3),:)).^2);
    TRI_LADOS(i,2)=lados(k);
    k=k+1;
    lados(k)=sum((vertices(TRI(i,2),:)-vertices(TRI(i,3),:)).^2);
    TRI_LADOS(i,3)=lados(k);
    maxlad(i)=max(TRI_LADOS(i,:));
end

TRI_LADOS=sqrt(TRI_LADOS);
lados=sqrt(lados);
maxlad=sqrt(maxlad);

end