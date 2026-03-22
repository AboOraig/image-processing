%
% matrice(Ni,Nj)->Tenseur(ni,nj,nk)
% Ni,Nj dimensions de la matrice I initiale
% ni,nj dimensions d'une imagette
% nk    nombre d'imagettes = nik*njk
%              nik = nombre d'imagettes par colonne 
%              njk = nombre d'imagettes par ligne

function [T]=func2D_3D(I,ni,nj)
%I=[11,12,13,14;21,22,23,24;31,32,33,34;41,42,43,44];
%I=reshape((1:2*2*3*5)',2*3,2*5)
[Ni,Nj]=size(I);            % 
nik=floor(Ni/ni);           %
njk=floor(Nj/nj);           %

Ni=nik*ni;                  %  creation d'une sous-image II
Nj=njk*nj;                  %  de dimensions égales a un multiple
nk=nik*njk;                 %  entier d'imagettes
II=I(1:nik*ni,1:njk*nj);    %

indi=1:(nik);            %
indj=1:(njk);            %
ii=indi'*ones(1,njk);    % tableau d'indices colonnes
ij=ones(nik,1)*indj;     % tableau d'indices lignes
%k=ii+(ij-1)*nik;
ii=ii(:);
ij=ij(:);

for k=1:nk;
    T(:,:,k)=II((1:ni)+(ii(k)-1)*ni,(1:nj)+(ij(k)-1)*nj);
end
