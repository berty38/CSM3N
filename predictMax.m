function y = predictMax(x, n, k)

% given a point in local marginal polytope, predict highest scoring labels
% for each instance

y = zeros(n,1);
for i = 1:n
    probs = x(localIndex(i,1:k,n));
    [~, y(i)] = max(probs);
end
