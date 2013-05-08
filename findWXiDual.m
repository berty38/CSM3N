function [w, xi, obj] = findWXiDual(Y, featureMap, labels, C)

% for vanilla LP-inference max margin

options = optimset('display', 'off');

options.MSK_IPAR_INTPNT_NUM_THREADS = 4;

X = bsxfun(@minus, featureMap * Y, featureMap * labels);
K = X'*X;

A = ones(1,size(Y,2));
b = C;
f = - sum(abs(bsxfun(@minus, Y, labels)))';

[alpha, obj, status] = quadprog(K, f, A, b, [], [], zeros(size(f)), [], [], options);

obj = - obj;

if isempty(alpha)
    w = [];
    xi = [];
    obj = [];
    return;
end

w = -X * alpha;

sv = alpha > C * 1e-6;

if nnz(sv) == 0
    [~, ind] = max(alpha);
    sv(ind) = true;
end

xi = mean(X(:, sv)'*w - f(sv));
