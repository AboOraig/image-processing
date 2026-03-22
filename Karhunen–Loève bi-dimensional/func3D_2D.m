%
% Tenseur(ni,nj,nk)->matrice(Ni,Nj)
% Ni,Nj dimensions de la matrice I initiale
% ni,nj dimensions d'une imagette
% nk    nombre d'imagettes = nik*njk
%              nik = nombre d'imagettes par colonne
%              njk = nombre d'imagettes par ligne
function [M]=func3D_2D(T,Ni,Nj)
M=zeros(Ni,Nj);
%I=[11,12,13,14;21,22,23,24;31,32,33,34;41,42,43,44];
%I=reshape((1:2*2*3*5)',2*3,2*5)
[ni,nj,nk]=size(T);

nik=floor(Ni/ni);
njk=floor(Nj/nj);
Ni=nik*ni;
Nj=njk*nj;
nk=nik*njk

indi=1:(nik); 
indj=1:(njk);
ii=indi'*ones(1,njk);    % tableau d'indices colonnes
ij=ones(nik,1)*indj;     % tableau d'indices lignes
k=ii+(ij-1)*nik;
ii=ii(:);
ij=ij(:);
for k=1:nk;
    M((1:ni)+(ii(k)-1)*ni,(1:nj)+(ij(k)-1)*nj)=T(:,:,k);
end