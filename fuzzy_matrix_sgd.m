clear; close all; rng(42);   % for reproducibility

%% 1. Parameters (same as paper’s 3x3 example)
n = 3;              % matrix dimension
R = 50;             % number of sample pairs
tau = 0.01;         % smoothing parameter
lr = 0.1;           % learning rate
epochs = 50;        % training epochs

%% 2. Generate synthetic data
% True fuzzy matrix (random in [0,1])
P_true = rand(n, n); %is a random n×n matrix

% Input vectors (R x n) uniformly distributed in [0,1]
S = rand(R, n); %Input vectors

% Target output vectors using exact max-min composition
T = zeros(R, n); %Target output
for i = 1:R
    for j = 1:n
        T(i,j) = max(min(S(i,:), P_true(:,j)'));
    end
end

%% 3. Define smooth max-min function and gradient
% Smooth approximation of t_j = max_k min(s_k, p_kj)
smooth_maxmin = @(s, P) tau * log( sum( 1 ./ (exp(-s/tau) + exp(-P/tau)) , 1) );

% Gradient of MSE loss w.r.t. matrix P (as derived in the paper, Eq. 20 corrected)
function grad = compute_gradient(s, t_true, P, tau)
    n = size(P, 1);
    % predicted output (smooth)
    t_pred = tau * log( sum(1 ./ (exp(-s/tau) + exp(-P/tau)), 1) );  % row vector
    
    % denominator sum for each column j
    denom = sum(1 ./ (exp(-s/tau) + exp(-P/tau)), 1);  % 1 x n
    
    grad = zeros(n);
    for i = 1:n
        for j = 1:n
            exp_neg_si = exp(-s(i)/tau);
            exp_neg_pij = exp(-P(i,j)/tau);
            term = exp_neg_si + exp_neg_pij;
            dtj_dpij = (1/denom(j)) * (1/term^2) * exp_neg_pij;
            grad(i,j) = (t_pred(j) - t_true(j)) * dtj_dpij;
        end
    end
end

%% 4. Training loop (SGD, one sample at a time)
P_learned = rand(n, n);   % initial matrix (random in [0,1])
mse_history = zeros(epochs, 1);
mre_history = zeros(epochs, 1);

for ep = 1:epochs
    % Shuffle sample order for stochastic updates
    idx = randperm(R);
    for k = idx
        s = S(k, :);        % 1 x n
        t = T(k, :);
        
        % Compute gradient and update
        grad = compute_gradient(s, t, P_learned, tau);
        P_learned = P_learned - lr * grad;
        
        % Project onto [0,1] to keep fuzzy matrix valid
        P_learned(P_learned < 0) = 0;
        P_learned(P_learned > 1) = 1;
    end
    
    % After each epoch, evaluate on all samples (using smooth predictions)
    T_pred = zeros(R, n);
    for i = 1:R
        T_pred(i,:) = smooth_maxmin(S(i,:), P_learned);
    end
    mse_history(ep) = mean( (T_pred - T).^2, 'all' );
    mre_history(ep) = norm(P_learned - P_true, 'fro');
    
    fprintf('Epoch %3d: MSE = %.2e, MRE = %.4f\n', ep, mse_history(ep), mre_history(ep));
end

%% 5. Display final results
disp('True matrix P_true:');
disp(P_true);
disp('Learned matrix P_learned:');
disp(P_learned);
disp('Difference (P_learned - P_true):');
disp(P_learned - P_true);

%% 6. Plots (similar to Figures 3–5 in the paper)
figure('Position', [100, 100, 1200, 400]);

% (a) Element-wise true vs. learned
subplot(1,3,1);
hold on;
for i = 1:n
    for j = 1:n
        plot([1, 2], [P_true(i,j), P_true(i,j)], 'r--', 'LineWidth', 1.2);
        plot(2, P_learned(i,j), 'bx', 'MarkerSize', 10, 'LineWidth', 1.5);
    end
end
xlim([0.5, 2.5]); ylim([0, 1]);
set(gca, 'XTick', [1, 2], 'XTickLabel', {'True', 'Learned'});
ylabel('Element value');
title('True (red dash) vs Learned (blue x)');
grid on;

% (b) MSE over epochs
subplot(1,3,2);
semilogy(1:epochs, mse_history, 'b-', 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('MSE');
title('Mean Squared Error');
grid on;

% (c) Reconstruction error over epochs
subplot(1,3,3);
plot(1:epochs, mre_history, 'r-', 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('Frobenius norm');
title('Matrix Reconstruction Error');
grid on;

sgtitle('Fuzzy Matrix Learning (n=3, R=50, \tau=0.01, \eta=0.1)');