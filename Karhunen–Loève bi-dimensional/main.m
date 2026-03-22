clc; clear all; close all;

%% 1.
% 1.1 Bruits blancs 2D
N = 256;
A1 = rand(N, N);     % uniforme
A2 = randn(N, N);    % gaussien

figure;
subplot(1,2,1); imshow(A1, []); title('Texture bruit blanc uniforme (rand)');
subplot(1,2,2); imshow(A2, []); title('Texture bruit blanc gaussien (randn)');

% 1.2 Masques 3x3 et filtrages
M  = ones(3);                               % isotrope
M1 = [0 0 1; 0 1 0; 1 0 0];                  % anisotrope
M2 = [1 0 0; 0 1 0; 0 0 1];                  % anisotrope

I1 = filter2(M,  A2, 'same');
I_x = filter2(M1, A2, 'same');
I_y = filter2(M2, A2, 'same');

% centrage + normalisation 
normu = @(X) (X - mean(X(:))) / sqrt(mean((X(:)-mean(X(:))).^2)+eps);
I1n = normu(I1); I_xn = normu(I_x); I_yn = normu(I_y);

figure;
subplot(1,3,1); imshow(I1n, []); title('Gaussien filtré par M (normalisé)');
subplot(1,3,2); imshow(I_xn, []); title('Gaussien filtré par M1 (normalisé)');
subplot(1,3,3); imshow(I_yn, []); title('Gaussien filtré par M2 (normalisé)');

% 1.3 Masque gaussien anisotrope et corrélé
p = 5;                     % demi-taille du masque -> L = 11
L = 2*p + 1;
a = 4; b = 4; r = 0.5;     % échelles et corrélation

[x0, y0] = meshgrid(-p:p, -p:p);

% Noyau gaussien 2D avec corrélation r
M_gauss = exp( - (1/(2*(1 - r^2))) * ( (x0.^2)/(a^2) + (y0.^2)/(b^2) - 2*r*(x0.*y0)/(a*b) ) );
M_gauss = M_gauss / sum(M_gauss(:));                 % normalisation en énergie

I_gauss = filter2(M_gauss, A2, 'same');

% Suppression des bords (p pixels de chaque côté)
I_gauss_crop = I_gauss(p+1:end-p, p+1:end-p);

% Normalisation (zéro-moyenne, puissance 1)
I_gauss_crop = normu(I_gauss_crop);

figure;
imshow(I_gauss_crop, []);
title(sprintf('Gaussien filtré par masque gaussien (L=%d, r=%.2f)', L, r));

% 1.4 Découpage en imagettes LxL -> matrice (L^2 x Nk)
% Si dimension pas multiple de L : tronquer proprement
[H,W] = size(I_gauss_crop);
Htrim = floor(H/L)*L;
Wtrim = floor(W/L)*L;
I_trim = I_gauss_crop(1:Htrim, 1:Wtrim);

% Extraction de blocs non chevauchés (distinct)
% -> chaque colonne est une imagette vectorisée (L^2 x 1)
C = im2col(I_trim, [L L], 'distinct');   % C : (L^2 x Nk)
Nk = size(C, 2);

fprintf('Nombre d\''imagettes extraites : %d (taille %dx%d chacune)\n', Nk, L, L);

% Reconstruire une pile 3D (L x L x Nk) pour visualiser quelques patches
C3 = reshape(C, [L, L, Nk]);             % (linéaire par colonnes)
figure;
for k = 1:min(9, Nk)
    subplot(3,3,k); imshow(C3(:,:,k), []); axis off
    title(sprintf('Patch #%d', k));
end
sgtitle('Exemples d''imagettes L x L');


%% 2.
% C : (L^2 x Nk) colonnes = imagettes vectorisées (si tu pars de C)
% On convertit en vecteurs LIGNE comme demandé par l'énoncé :
[L2, Nk] = size(C);                 % L2 = 121, Nk = nb d'imagettes
L = sqrt(L2); L = round(L);

% 2.1 Vecteurs ligne v_k (Nk x 121)
V = C.';                             % V : (Nk x L2), chaque LIGNE = v_k (1x121)

% 2.2 Matrice de covariance Gamma_V = (1/Nk) * Vc^T * Vc  (121 x 121)
mu = mean(V, 1);                     % (1 x L2)
Vc = V - mu;                         

Gamma_V = (Vc.' * Vc) / Nk;          % (L2 x L2)
Gamma_V = (Gamma_V + Gamma_V.')/2; 

% 2.3 Décomposition spectrale
[EigVec, EigVal] = eig(Gamma_V);         
lambda = real(diag(EigVal));
[lambda, idx] = sort(lambda, 'descend');
P = EigVec(:, idx);                      % base KLT (121 x 121)

% 2.4 Affichage des 4 premiers vecteurs propres sous forme d'imagettes 11x11
Kdisp = 4;
figure; tiledlayout(1,Kdisp,'TileSpacing','compact','Padding','compact');
for k = 1:Kdisp
    pk_img = reshape(P(:,k), [L, L]);    % (11 x 11)
    nexttile; imshow(pk_img, []); title(sprintf('Vecteur propre P_%d', k));
end

% Vérification MQ des projections = λ_k (comme au TP1)
alpha = Vc * P;                          % (Nk x L2) coeffs KL (lignes = réalisations)
MQ = mean(alpha.^2, 1);                  % (1 x L2)
figure; plot(1:L2, MQ, 'o-'); hold on; plot(1:L2, lambda, 'x--');
xlabel('k'); ylabel('Valeur'); grid on;
legend('MQ des projections','\lambda_k'); 
title('Vérification : MQ des projections = \lambda_k');


%% 3. 
% 3.1 
m_max = min(20, size(P,2));       % nombre max de composantes à tester
ind_img = 1;
v  = V(ind_img, :).';             % imagette choisie
v0 = v - mu.';                    % centré

% Boucle sur m = 1..m_max : projection, reconstruction, EQM
EQM_patch = zeros(m_max,1);       
EQM_theo  = zeros(m_max,1);      

for m = 1:m_max
    % Projection sur la base KLT
    Bm = P(:,1:m);                         
    coeff = Bm.' * v0;                        
    vhat = Bm * coeff + mu.';                

    % EQM de la réalisation (moyenne par pixel)
    EQM_patch(m) = mean((v - vhat).^2);

    % Erreur théorique KLT (moyenne par pixel)
    % E[||v - vhat||^2] = sum_{i>m} lambda_i
    EQM_theo(m)  = sum(lambda(m+1:end)) / L2;
end

figure();
plot(1:m_max, EQM_patch, 'o-', 'LineWidth', 1.2); 
hold on;
plot(1:m_max, EQM_theo,  'x--','LineWidth', 1.2);
grid on; xlabel('Nombre de composantes m'); 
ylabel('EQM (moyenne par pixel)');
legend('EQM réalisation','Somme des \lambda non retenus / L^2','Location','best');
title('Compression KLT : erreur réelle vs erreur théorique');


%% 4. 
% rng(0);

% Champ propre
I_clean = I_gauss_crop;

% Ajout d'un bruit blanc gaussien de puissance 1/4
sigma_b2 = 1/4;                    
B = sqrt(sigma_b2) * randn(size(I_clean));
I_noisy = I_clean + B;               

% Découpage en imagettes non chevauchées L x L -> (L^2 x Nk)
[H,W]  = size(I_clean);
Htrim  = floor(H/L)*L;  Wtrim = floor(W/L)*L;

Ic_trim = I_clean(1:Htrim, 1:Wtrim);
In_trim = I_noisy(1:Htrim, 1:Wtrim);

% Extraction via func2D_3D
T_clean = func2D_3D(Ic_trim, L, L);      % (L x L x Nk)
T_noisy = func2D_3D(In_trim, L, L);      % (L x L x Nk)

[L2, L2, Nk] = size(T_clean);
fprintf('Nb imagettes (propres/bruitées) : %d (taille %dx%d)\n', Nk, L, L);

% Vectorisation en LIGNE (Nk x L^2)
V_clean = reshape(T_clean, [L*L, size(T_clean,3)]).';
V_noisy = reshape(T_noisy, [L*L, size(T_noisy,3)]).';

% Centrage séparé et covariances
mu_clean = mean(V_clean, 1);
mu_noisy = mean(V_noisy, 1);

Vc_clean = V_clean - mu_clean;
Vc_noisy = V_noisy - mu_noisy;

Gamma_clean = (Vc_clean.' * Vc_clean) / Nk;   Gamma_clean = (Gamma_clean+Gamma_clean')/2;
Gamma_noisy = (Vc_noisy.' * Vc_noisy) / Nk;   Gamma_noisy = (Gamma_noisy+Gamma_noisy')/2;

% Décompositions propres
[Ec, Dc] = eig(Gamma_clean);  lc = real(diag(Dc)); [lc, ic] = sort(lc, 'descend'); Pc = Ec(:, ic);
[En, Dn] = eig(Gamma_noisy);  ln = real(diag(Dn)); [ln, in] = sort(ln, 'descend'); Pn = En(:, in);

% Comparaison des spectres
figure();
plot(lc, 'o-', 'LineWidth', 1.2); 
hold on;
plot(ln, 'x--','LineWidth', 1.2);
grid on; xlabel('Indice k'); ylabel('\lambda_k');
legend('\lambda_k (propre)', '\lambda_k (bruité)', 'Location', 'northeast');
title('Champ propre vs champ bruité : valeurs propres');

% Affichage des 4 premiers vecteurs propres (propres vs bruités)
Kdisp = 4;
figure();
tiledlayout(2, Kdisp, 'TileSpacing','compact','Padding','compact');
for k = 1:Kdisp
    nexttile; imshow(reshape(Pc(:,k),[L L]), []); title(sprintf('Propre P_%d',k));
end
for k = 1:Kdisp
    nexttile; imshow(reshape(Pn(:,k),[L L]), []); title(sprintf('Bruité P''_%d',k));
end

% Similarité des k premiers vecteurs propres (cosinus d'angle, insensible au signe)
sim = zeros(Kdisp,1);
for k = 1:Kdisp
    sim(k) = abs(Pc(:,k)' * Pn(:,k));   % Pc/Pn sont orthonormés -> cos(angle)
end
fprintf('Similarité |<P_k, P''_k>| pour k=1..%d :\n', Kdisp);
disp(sim.');
