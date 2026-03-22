clc; clear all; close all;

img = rgb2gray(imread('lena_std.tif')); 
img = im2double(img);

figure;
subplot(231)
imshow(img, []); 
title('Image originale');
colormap('gray')
colorbar();

% Ajouter le flou
PSF = fspecial('motion', 35, 0);        % deplacement horiz longeur 35 
img_flou = imfilter(img, PSF, 'conv','circular');
subplot(232);
imshow(img_flou, []); 
title('Image Flou');
colorbar();

% Ajouter le bruit : gaussien 
v = 1e-5;
% bruit = a * randn(); % moy = 0, var = a^2
% img_bruit = img_flou + bruit;
% ou :  
img_bruit = imnoise(img_flou, "gaussian", 0, v);
subplot(233);
imshow(img_bruit, []); 
title('Image Bruit');
colorbar();


%% 1. Filtre inverse
nsr = 0;
image_res1 = deconvwnr(img_bruit, PSF, nsr);
subplot(234);
imshow(image_res1, [])
title('Resturation avec Filtre Inverse (NSR = 0)')
colormap('gray')
colorbar();


%% 2.1. Filtre Wiener simplifié : 1er boucle
% On cree une vecteur de K de 0 avec un bas de delta pour trouver la valeur 
% optimal et pour chaque valeur on trace l'erreur entre l'estimation et l'image
delta = 1e-5;                  
nsr_vals = 0:delta:delta*1000;
var_e = zeros(size(nsr_vals));     
diff = zeros(size(nsr_vals));     

for i = 1:length(nsr_vals)
    % Ici on n'utilise pas la diffrence entre l'image originale et l'image 
    % estimée car en cas reel on ne sait pas l'image originale.
    nsr = nsr_vals(i);
    image_res2 = deconvwnr(img_bruit, PSF, nsr);
    image_res2_flou = imfilter(image_res2, PSF, 'conv','circular');
    r = img_bruit(:) - image_res2_flou(:);
    var_e(i) = norm(r(:))^2 / (numel(img));
    diff(i) = abs(var_e(i) - v);
end

[~, idx_min] = min(diff);
nsr_opt = nsr_vals(idx_min);

disp(['Valeur de K estimée par 1er boucle : ', num2str(nsr_opt)]);
disp(['Valeur de K réelle : ', num2str(v / var(img(:)))]);

nsr = nsr_opt;
image_res2 = deconvwnr(img_bruit, PSF, nsr);
subplot(235);
imshow(image_res2, []);
title(['Resturation avec Filtre Wiener simplifié - 1er boucle (NSR = ',num2str(nsr),')']);
colormap('gray');
colorbar();

% figure;
% plot(nsr_vals, var_e);
% hold on;
% plot(nsr_vals, diff);
% xlabel('NSR (k)');
% ylabel('EQM');
% title('Erreur en fonction de NSR');


%% 2.2. Filtre Wiener simplifié : 2eme boucle
% Convergance avec dichotomie de K pour Filtre Wiener simplifié
% k0 -> k0 + pas : var_e - var_bruit > 0 : +/- pas
% pas = pas / 2 : var_e - var_bruit < 0 : +/- pas

K0 = 1e-4;
tol = v;           
maxIter = 100;

for iter = 1:maxIter

    image_res3 = deconvwnr(img_bruit, PSF, K0);
    image_res3_flou = imfilter(image_res3, PSF, 'conv','circular');

    r = img_bruit(:) - image_res3_flou(:);
    var_e = norm(r(:))^2 / numel(img);
    f = var_e - v;

    if abs(f) < tol
        break;
    end

    if f > 0
        K0 = K0 + delta;
    else
        K0 = K0 - delta;
        delta = delta / 2;
    end
end

nsr_opt = K0;
disp(['Valeur de K estimée par dichotomie : ', num2str(nsr_opt)]);
disp(['Valeur de K réelle : ', num2str(v / var(img(:)))]);

image_res3 = deconvwnr(img_bruit, PSF, nsr_opt);
subplot(236);
imshow(image_res3, []);
title(['Resturation avec Filtre Wiener simplifié - 2eme boucle (NSR = ', num2str(nsr_opt), ')']);
colormap('gray');
colorbar();


%% 3. Filtre Wiener : avec K reel 
nsr = v / var(img(:));
image_res4 = deconvwnr(img_bruit, PSF, nsr);

figure;
subplot(121);
imshow(image_res4, []),
title(['Resturation avec Filtre Wiener (simplifié) avec K réel (NSR = ',num2str(nsr),')']);
colormap('gray');
colorbar();


%% 3.1. Filtre Wiener : avec autocorrulation
% Estimation de autocorrulation pour l'image initiale et le bruit :
% fftn => model carre => inverse :
% ifftn(abs(fftn(...)).^2) mais on n'interse que de partie reel car notre
% image (function) est réel et on utilise fftshift pour avoir le max en centre.

F_img = fftn(img);
R_f = fftshift(real(ifftn(abs(F_img).^2)));

bruit = img_bruit - img_flou;
F_bruit = fftn(bruit);
R_n = fftshift(real(ifftn(abs(F_bruit).^2)));

image_res5 = deconvwnr(img_bruit, PSF, R_n, R_f);

subplot(122);
imshow(image_res5, []); 
title('Resturation avec Filtre Wiener (icorr & ncorr)');
colormap('gray');
colorbar();


%% 3.2. Filtre Wiener 
% normalement k(u,v) = S_nn(u,v)/S_ff(u,v) donc dans le cas reel on n'est pas l'image originale donc on ne peux pas avoir S_ff 
% on peut plus ou moins trouver S_nn(u,v) par observer une partie de l'image bruitee laquelle on peut dire que le bruit est uniforme 
% donc on doit creer un algo d'estimation qui converge vers cette valeur de K
% on va utiliser le meme boucle de K simplifie ! et on trace equelment l'erreur et prendre le min

maxIter = 30;
tol = v;           
eps0 = 1e-12;

% on utilise une zone de bruit pur : utiliser cette zone pour estimer S_n
figure; imshow(img_bruit, []); 
title('Sélectionnez une zone de bruit uniforme');
% Dessiner un rectangle à la souris (déplace/resize puis double-clic pour valider)
h = drawrectangle('Label','ROI bruit','Color','y');

% Récupérer les coordonnées entières (x,y,width,height)
pos = round(h.Position);      % [x y w h]
x1 = max(1, pos(1));
y1 = max(1, pos(2));
x2 = min(size(img_bruit,2), x1 + pos(3) - 1);
y2 = min(size(img_bruit,1), y1 + pos(4) - 1);

% Patch bruit (approx) = zone bruitée - zone floue (si img_flou connue)
patch = img_bruit(y1:y2, x1:x2) - img_flou(y1:y2, x1:x2);

% Spectre de puissance du bruit remis à la taille de l'image
S_n = abs(fftn(patch, size(img_bruit))).^2;

Y = fftn(img_bruit);
S_y = abs(Y).^2;
S_f = max(S_y - S_n, eps0);

prev_f = zeros(size(img_bruit));
H = psf2otf(PSF, size(img_bruit));
for it = 1:maxIter
    NSR_mat = S_n ./ max(S_f, eps0);

    f_est = deconvwnr(img_bruit, PSF, NSR_mat);

    % calcul du résidu dans l'espace image
    y_hat = imfilter(f_est, PSF, 'conv', 'circular');
    resid = img_bruit - y_hat;

    % estimer S_n à partir du résidu (periodogramme lissé)
    Rn = abs(fftn(resid)).^2;
    % lissage optionnel (p.ex. moyennage locale) pour stabiliser
    S_n = imgaussfilt(real(Rn), 1);  % lissage gaussien du periodogramme

    % estimer S_f à partir de la restauration
    Sf_new = abs(fftn(f_est)).^2;
    % on peut stabiliser / mélanger avec l'estimation précédente
    alpha = 0.6;
    S_f = alpha * Sf_new + (1-alpha) * S_f;
    S_f = max(S_f, eps0);

    % critère de convergence (norme relative sur f_est)
    rel_change = norm(f_est(:)-prev_f(:)) / (norm(prev_f(:))+eps0);
    if rel_change < tol
        fprintf('Converged after %d iter (rel_change=%.2e)\n', it, rel_change);
        break;
    end
    prev_f = f_est;
end

image_res_wiener_iter = f_est;
figure;
imshow(image_res_wiener_iter, []);
title('Resturation avec Filtre Wiener (S_n et S_f)');
