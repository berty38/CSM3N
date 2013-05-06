function violation = checkConstraints(Y, featureMap, labels, w, xi, kappa)

d = size(featureMap,1);

lb = -inf(d + 1, 1);
lb(d + 1) = 0;
ub = [];

A = zeros(size(Y,2), d+1);
b = zeros(size(Y,2),1);

for i = 1:size(Y,2)
    y = Y(:,i);
    newA = [featureMap * y - featureMap * labels; -1];
    newB = (y' * y - labels' * labels) * 0.5 * kappa - sum(abs(y - labels));
    A(i,:) = newA';
    b(i) = newB';
end

x = [w; xi];


violation = A*x - b;

