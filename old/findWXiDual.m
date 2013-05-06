function [w, xi, obj] = findWXiDual(Y, kappa, featureMap, labels, C)

options = optimset('display', 'off');

options.MSK_IPAR_INTPNT_NUM_THREADS = 4;

X = bsxfun(@minus, featureMap * Y, featureMap * labels);
% if kappa == 0
K = kappa * X'*X;

% else
%     K = kappa * X'*X;
% end
A = ones(1,size(Y,2));
b = C;
f = 0.5 * kappa * (sum(Y.^2) - labels'*labels)' - sum(abs(bsxfun(@minus, Y, labels)))';

% [V,D] = eig(K);
% while min(diag(D)) < 0
%     fprintf('Adding diagonal to kernel of magnitude %f\n', min(diag(D))*10);
%     K = K - min(diag(D)) * 10 * eye(size(K));
%     [V,D] = eig(K);
%     [alpha, obj, status] = quadprog(K, f, A, b, [], [], zeros(size(f)), [], [], options);
% end


[alpha, obj, status] = quadprog(K, f, A, b, [], [], zeros(size(f)), [], [], options);

obj = - obj;

if isempty(alpha)
    w = [];
    xi = [];
    obj = [];
    return;
end

% if kappa == 0
w = -X * alpha * kappa;
% else
%     w = -X * alpha * kappa;
% end

sv = alpha > C * 1e-6;

if nnz(sv) == 0
    [~, ind] = max(alpha);
    sv(ind) = true;
end

xi = mean(X(:, sv)'*w - f(sv));
