clc; clear all; close all;

%% 1.
% 1.1.
N = 4096;
U = randn(1, N);
delta_t = 1;
fe = 1/delta_t; 
fshift_U = (0:N-1)*(fe/N);
fshift_T = (-N/2:N/2-1)*(fe/N);

% Si t est du temps (ex. secondes), alors fe vaut en Hz (1 cycle par seconde).
% Si t représente une distance, alors fe vaut m⁻¹, c’est une fréquence spatiale.

% 1.2.
% La TF d’un signal échantillonné est périodique en fréquence.

% 1.3.
n = 32; 
fcb = (1/3) * (fe/2); 
fch = (2/3) * (fe/2);    
h = fir1(n-1, [fcb fch]/(fe/2), 'bandpass');  % normalisation

figure;
freqz(h, 1, 1024, fe);
title('Filtre passe-bande FIR (fir1) H1');

% 1.4.
T = filter(h, 1, U); 
T = T - mean(T);         % centrage
P = mean(T.^2);          % puissance moyenne
T = T / sqrt(P);         % normalisation de puissance

fprintf('Puissance après normalisation : %.3f\n', mean(T.^2));

U_fft = abs(fftshift(fft(U)));
T_fft = abs(fftshift(fft(T)));

figure;
subplot(2,1,1);
plot(fshift_U, U_fft);
xlabel('Fréquence normalisée');
ylabel('|FFT(U)|');
title('Spectre du signal original U');
grid on;

subplot(2,1,2);
plot(fshift_U, T_fft);
xlabel('Fréquence normalisée');
ylabel('|FFT(T)|');
title('Spectre du signal filtré et normalisé T');
grid on;

% 1.5.
L = 16;
num_segments = N/L;
S = reshape(T, L, num_segments);

% 1.6.
Pxx = zeros(size(S));
for k = 1:num_segments
    sk = S(:,k);
    Pxx(:,k) = (1/L) * abs(fft(sk)).^2;  
end

Pxx_mean = mean(Pxx, 2);  
f = (0:L-1)*(fe/L);

figure;
plot(f, fftshift(Pxx_mean));
xlabel('Fréquence');
ylabel('DSP moyenne');
title('Densité de puissance moyenne du processus S');
grid on;

%% 2.
% 2.1.
M = zeros(L, L);
for k = 1:num_segments
    sk = S(:,k);
    M = M + (sk * sk');     
end
M = M / num_segments;

% 2.2.
[V, D] = eig(M); % V = vecteurs propres, D diagonal = valeurs propres
valp = diag(D);
[valp_sorted, idx] = sort(valp, 'descend');
V_sorted = V(:, idx);
P = V_sorted(:,1);             
lambda = valp_sorted(1);     
% P = P / norm(P); % normalisation

% 2.3.
MQ = [];
for k = 1:L
    Pk = V_sorted(:,k);                 
    alpha_k = Pk' * S; % projections de tous les s_k
    mq = mean(alpha_k.^2);     
    MQ = [MQ mq];
    fprintf('λk = %.6f, MQ des projections = %.6f\n', valp_sorted(k), mq);
end

figure;
plot(1:L, MQ, 'o-');
hold on;
plot(1:L, valp_sorted, 'x--');
xlabel('Composante principale k');
ylabel('Valeur / MQ');
legend('MQ des projections','Valeurs propres λ_k');
title('Vérification : MQ des projections = λ_k');
grid on;


%% 3.
P_old = P / norm(P);
eps_values = linspace(0, fcb*0.9, 20);
% eps_values = [0, 1e-3, 5e-3, 1e-2, 2e-2];
lambda_vals = zeros(size(eps_values));
mq_on_Pold_vals = zeros(size(eps_values));
diff_vals = zeros(size(eps_values));

for i = 1:length(eps_values)
    epsi = eps_values(i);

    fcb2 = max(0, fcb - epsi);
    fch2 = min(fe/2, fch + epsi);
    h2 = fir1(n-1, [fcb2 fch2]/(fe/2), 'bandpass');
    T2 = filter(h2, 1, U);
    T2 = T2 - mean(T2);
    T2 = T2 / sqrt(mean(T2.^2));

    % Segmentation
    S2 = reshape(T2, L, num_segments);

    % Matrice de covariance empirique
    M2 = zeros(L,L);
    for k = 1:num_segments
        sk = S2(:,k);
        M2 = M2 + (sk * sk');
    end
    M2 = M2 / num_segments;

    % Valeurs propres
    [V2, D2] = eig(M2);
    valp2 = diag(D2);
    [valp2_sorted, idx2] = sort(valp2, 'descend');
    V2_sorted = V2(:, idx2);

    % Plus grande valeur propre
    lambda_max = valp2_sorted(1);

    % MQ sur le vecteur propre du premier processus
    alpha_on_Pold = P_old' * S2;
    mq_on_Pold = mean(alpha_on_Pold.^2);

    lambda_vals(i) = lambda_max;
    mq_on_Pold_vals(i) = mq_on_Pold;
    diff_vals(i) = lambda_max - mq_on_Pold;

    fprintf('eps=%.5f : lambda_max=%.6f, MQ_on_Pold=%.6f, diff=%.6f\n', ...
        epsi, lambda_max, mq_on_Pold, diff_vals(i));
end

figure;
plot(eps_values, lambda_vals, 'o-', eps_values, mq_on_Pold_vals, 'x--', 'LineWidth', 1.2);
xlabel('\epsilon (Hz)');
ylabel('Valeur');
legend('\lambda_{max}(M2)','MQ_{S2}(P_{old})','Location','best');
title('Comparaison λ_{max} et MQ des projections sur P_{old}');
grid on;

figure;
plot(eps_values, diff_vals, 's-', 'LineWidth', 1.2);
xlabel('\epsilon (Hz)');
ylabel('\lambda_{max} - MQ_{S2}(P_{old})');
title('Différence entre λ_{max} et MQ des projections sur P_{old}');
grid on;
