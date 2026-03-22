clc; clear all; close all;

%% 1.
I1 = imread("Texture1.jpg");
I1 = im2double(I1);    % normalisation [0,1]

figure;
subplot(1,2,1); 
imshow(I1, []);
title('Texture d''intérêt (Texture1.jpg)');

I1 = I1 - mean(I1(:));
I1 = I1 / sqrt(mean(I1(:).^2)); % puissance unitaire

subplot(1,2,2); 
imshow(I1, []);
title('Apres centrage et normalisation');

% Extraction des imagettes non chevauchées
L = 11;                % taille d'une imagette
C1 = im2col(I1, [L L], 'distinct');  % (L^2 x Nk)
[L2, Nk] = size(C1);
fprintf('Nombre d''imagettes extraites : %d (taille %dx%d)\n', Nk, L, L);

% Mise en forme (vecteurs lignes, centrage)
V1 = C1.';               % (Nk x L^2)
mu1 = mean(V1, 1);
V1c = V1 - mu1;

% Matrice de covariance S1
S1 = (V1c.' * V1c) / Nk;
S1 = (S1 + S1.') / 2;    % symétrisation numérique
fprintf('Matrice S1 : %dx%d\n', size(S1,1), size(S1,2));

% Décomposition propre
[P1, D1] = eig(S1);
lambda1 = real(diag(D1));
[lambda1, idx] = sort(lambda1, 'descend');
P1 = P1(:, idx);
 
figure;
plot(lambda1, 'o-','LineWidth',1.2);
xlabel('Indice k'); 
ylabel('\lambda_k');
title('Spectre des valeurs propres de S_1');
grid on;
 
% Kdisp = 4;
% figure;
% tiledlayout(1, Kdisp, 'TileSpacing','compact','Padding','compact');
% for k = 1:Kdisp
%     nexttile;
%     imshow(reshape(P1(:,k), [L L]), []);
%     title(sprintf('Vecteur propre P_{1,%d}',k));
% end


%% 2.

P_vec = P1(:,1);                    % vecteur propre associé à λ_max
P_filt = reshape(P_vec, [L L]);     % imagette 11x11 pour filtrage

% figure;
% imshow(P_filt, []);
% title('Filtre optimal P (vecteur propre principal)');

% Image bruitée I1 (texture dans bruit microscopique)
I1_bruit = imread('Bruit_T1.jpg');
I1_bruit = im2double(I1_bruit);

figure;
subplot(1, 3, 1);
imshow(I1_bruit, []);
colormap('gray'); colorbar;
title('Image I_1 : texture dans un bruit à corrélation microscopique');

% % Centrage / normalisation 
I1_bruit = I1_bruit - mean(I1_bruit(:));
I1_bruit = I1_bruit / sqrt(mean(I1_bruit(:).^2) + eps);
 
% Filtrage de l'image par le filtre P (filtre adapté)
I1F = filter2(P_filt, I1_bruit);

subplot(1, 3, 2);
imshow(I1F, []);
colormap('gray'); colorbar;
title('Image filtrée I_{1F} = filter2(P, I_1)');

% Image des puissances locales (carrés des coefficients filtrés)
I1P = I1F.^2;                      % puissance locale (corrélation au carré)
I1P_norm = I1P / max(I1P(:));      % normalisation pour affichage
mean_power = mean(I1P(:));         % E{I1F^2} empirique

fprintf('Espérance empirique des carrés : E{I1F^2} = %.4f\n', mean_power);

subplot(1, 3, 3);
imshow(I1P_norm, []);
colormap('gray'); colorbar;
title('Image des puissances locales I_{1P} = I_{1F}^2');


%% 3.
% 3.1 Calcul de la matrice de covariance S2
I2_tex = imread('Texture2.jpg');
I2_tex = im2double(I2_tex);

figure;
subplot(1,2,1); 
imshow(I2_tex, []);
title('Texture d''intérêt (Texture2.jpg)');

I2_tex = I2_tex - mean(I2_tex(:));
I2_tex = I2_tex / sqrt(mean(I2_tex(:).^2));

subplot(1,2,2); 
imshow(I2_tex, []);
title('Apres centrage et normalisation');

C2 = im2col(I2_tex, [L L], 'distinct');    % (L^2 x Nk2)
[L2, Nk2] = size(C2);
V2  = C2.';                                % (Nk2 x L^2)
mu2 = mean(V2, 1);
V2c = V2 - mu2;                            % centrage
fprintf('Nombre d''imagettes extraites : %d (taille %dx%d)\n', Nk, L, L);

% matrice de covariance S2
S2 = (V2c.' * V2c) / Nk2;
S2 = (S2 + S2.')/2;                        % symétrisation

fprintf('Matrice S2 : %dx%d\n', size(S2,1), size(S2,2));

% Décomposition propre
[P2, D2] = eig(S2);
lambda2 = real(diag(D2));
[lambda2, idx] = sort(lambda2, 'descend');
P2 = P2(:, idx);

figure;
plot(lambda2, 'o-','LineWidth',1.2);
xlabel('Indice k'); 
ylabel('\lambda_k');
title('Spectre des valeurs propres de S_2');
grid on;

Kdisp = 4;
figure;
tiledlayout(1, Kdisp, 'TileSpacing','compact','Padding','compact');
for k = 1:Kdisp
    nexttile;
    imshow(reshape(P2(:,k), [L L]), []);
    title(sprintf('Vecteur propre P_{2,%d}',k));
end

% 3.2 Filtre maximisant le SNR entre texture1 et texture2
% on maximise J(p) = (p^T S1 p)/(p^T S2 p)
% -> problème de valeurs propres généralisées : S1 p = lambda S2 p

eps_reg = 1e-3;                            % petite régularisation
S2_reg = S2 + eps_reg*eye(size(S2));

[Vec_g, Val_g] = eig(S1, S2_reg);          % valeurs propres généralisées
lambda_g = real(diag(Val_g));
[lambda_g, idx_g] = sort(lambda_g, 'descend');
p_opt = Vec_g(:, idx_g(1));                % vecteur propre associé à λ_max
P_opt = reshape(p_opt, [L L]);             % filtre 2D LxL

figure;
imshow(P_opt, []);
title('Filtre optimal (S1 vs S2)');

% 3.3 Filtrage de l'image mélange I2

I2 = imread('Texture2_T1.jpg');            % texture1 dans texture2
I2 = im2double(I2);

figure;
subplot(1,3,1);
imshow(I2, []);
colormap('gray'); colorbar;
title('Image I_2 : texture1 dans texture2');

% filtrage 
I2F = filter2(P_opt, I2);

subplot(1,3,2);
imshow(I2F, []);
colormap('gray'); colorbar;
title('Image filtrée I_{2F}');

% puissances locales
I2P = I2F.^2;
I2P_norm = I2P / max(I2P(:));              % normalisation pour affichage

subplot(1,3,3);
imshow(I2P_norm, []);
colormap('gray'); colorbar;
title('Puissances locales normalisées I_{2P}');

% 3.4 Seuillage pour détection (s = 0.2 et s = 0.05)
s_vals = [0.2 0.05];
figure;
for k = 1:numel(s_vals)
    s = s_vals(k);
    I2s = I2P_norm >= s;                   % carte binaire
    subplot(1, numel(s_vals), k);
    imshow(I2s, []);
    title(sprintf('Image seuillée, s = %.2f', s));
end
sgtitle('Détection de la texture 1 dans la texture 2');


%% 4. Courbes C.O.R (ROC)
% 4.1 Chargement du masque "vérité terrain"
M = imread('Imsansbruit.bmp');   % masque binaire (0/255 ou 0/1)
M = im2double(M);

% Nombre total de pixels "signal présent" et "signal absent"
N_pos = sum(M(:));           % vrais "1"
N_neg = sum(~M(:));          % vrais "0"

fprintf('Nb pixels signal présent (M=1) : %d\n', N_pos);
fprintf('Nb pixels signal absent  (M=0) : %d\n', N_neg);

% 4.2 Cartes de puissance pour les deux filtres
% (a) Filtre optimal S1 vs S2 (déjà calculé à la partie 3)
% (b) Second filtre de comparaison : vecteur propre de S1 (partie 2)
P_filt = reshape(P1(:,1), [L L]);     % premier vecteur propre de S1
I2F_alt = filter2(P_filt, I2);
I2P_alt = I2F_alt.^2;
I2P_alt_norm = I2P_alt / max(I2P_alt(:));

% 4.3 Balayage des seuils et calcul (Pd, Pfa)
t = 0:0.1:10; 
seuils = (exp(t) - 1) / (exp(10) - 1);   % dans [0,1]

Pd_opt  = zeros(size(seuils));
Pfa_opt = zeros(size(seuils));
Pd_alt  = zeros(size(seuils));
Pfa_alt = zeros(size(seuils));

for k = 1:numel(seuils)
    s = seuils(k);

    % filtre optimal
    Det_opt = I2P_norm >= s;    % carte binaire de détection

    TP = sum( Det_opt(:)  &  M(:));   % bonnes détections
    FP = sum( Det_opt(:)  & ~M(:));   % fausses alarmes
    FN = sum(~Det_opt(:)  &  M(:));   % non-détectés
    TN = sum(~Det_opt(:)  & ~M(:));   % bonnes absences

    Pd_opt(k)  = TP / (TP + FN);          % probabilité de détection
    Pfa_opt(k) = FP / (FP + TN);          % probabilité de fausse alarme

    % filtre alternatif (P_filt)
    Det_alt = I2P_alt_norm >= s;

    TP = sum( Det_alt(:)  &  M(:));
    FP = sum( Det_alt(:)  & ~M(:));
    FN = sum(~Det_alt(:)  &  M(:));
    TN = sum(~Det_alt(:)  & ~M(:));

    Pd_alt(k)  = TP / (TP + FN);
    Pfa_alt(k) = FP / (FP + TN);
end

% 4.4 Tracé des courbes C.O.R
figure;
plot(Pfa_opt, Pd_opt, 'o-', 'LineWidth', 1.5); 
hold on;
plot(Pfa_alt, Pd_alt, 'x--', 'LineWidth', 1.5);
grid on;
xlabel('Probabilité de fausse alarme P_{fa}');
ylabel('Probabilité de détection P_d');
title('Courbes C.O.R pour deux filtres');
legend('Filtre optimal S_1 / S_2', 'Filtre P_1 (texture 1 seule)', ...
       'Location', 'SouthEast');
axis([0 1 0 1]);