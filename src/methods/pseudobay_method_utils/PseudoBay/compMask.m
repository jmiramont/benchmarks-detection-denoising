function [ Mask_out ] = compMask( tf,Pnei,N,sumM )
% [ Mask_out ] = compMask( tf,Pnei,N )
%
% Compute the mask for signal reconstruction
%
% 
% INPUT:
% tf         : ridges position
% Pnei       : mask width
% N          : number of frequency bin
%
% OUTPUT:
% Mask_out   : binary mask
%
% Author: Q.Legros (quentin.legros@telecom-paris.fr) and D.Fourer
% Date: 1-mar-2021


[Niter,Ncomp]=size(tf);

% Computation of the mask
Mask_out=zeros(N*Niter,Ncomp);
veccol = transpose((1:Niter)-1).*N;
for Nc = 1:Ncomp
    Mask_out(max(tf(:,Nc)+veccol-1,ones(Niter,1)),Nc)=1;
    for pn = 1:Pnei
        Mask_out(max(tf(:,Nc)+veccol-1-pn,ones(Niter,1)),Nc)=1;
        Mask_out(min(tf(:,Nc)+veccol-1+pn,(N*Niter)*ones(Niter,1)),Nc)=1; 
    end
end

Mask_out = reshape(Mask_out,[N,Niter,Ncomp]);

if Ncomp < 2
   Mask_out = [Mask_out;Mask_out(end:-1:1,:)];
else
   mask2 = zeros(2*N,Niter,Ncomp);
   for i = 1:Ncomp
     mask2(:,:,i) = [Mask_out(:,:,i);Mask_out(end:-1:1,:,i)];
   end 
   Mask_out = mask2;
end
    
if sumM~=0
    Mask_out = sum(Mask_out,3);
end


end
