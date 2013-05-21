function [cm, acc, f1, f1class] = singletonStats(y, p, k, weighted)

% Computes the confusion matrix of the singleton cliques.
% y - (nk x 1) ground truth in {0,1}
% p - (nk x 1) predictions in [0,1] (simplex constraints on cliques)
% k - number of classes
% weighted - (optional) whether to compute weighted or unweighted stats
% cm - (k x k) confusion matrix, where rows are true labels
% acc - (weighted) accuracy
% f1 - (weighted) F1

if nargin < 4
	weighted = 1;
end

assert(~isempty(y))
assert(length(y) == length(p))

% Confusion matrix
cm = zeros(k);
for c=1:k:length(y)
	y_c = find(y(c:c+k-1));
	[~,p_c] = max(p(c:c+k-1));
	cm(y_c,p_c) = cm(y_c,p_c) + 1;
end

% Accuracy
acc = sum(diag(cm)) / sum(cm(:));

% F1
pre = diag(cm) ./ sum(cm,1)';
pre(isnan(pre)) = 1;
rec = diag(cm) ./ sum(cm,2);
rec(isnan(rec)) = 1;
f1class = 2 * (pre .* rec) ./ (pre + rec);
f1class(isnan(f1class)) = 0;
classdist = sum(cm,2).^-1;
classdist(isinf(classdist)) = 0;
f1 = f1class' * classdist;

