function [w, xi, obj] = findWXiPrimal(Y, featureMap, labels, C)

d = size(featureMap,1);

f = zeros(d + 1, 1);
f(d + 1) = C;

H = speye(d + 1);
H(d + 1, d + 1) = 0;


Aeq = []; beq = [];
lb = -inf(d + 1, 1);
lb(d + 1) = 0;
ub = [];

options = optimset('algorithm', 'active-set', 'display', 'off'); % for MATLAB
options.MSK_IPAR_INTPNT_NUM_THREADS = 4; % for MOSEK

A = zeros(size(Y,2), d+1);
b = zeros(size(Y,2),1);

for i = 1:size(Y,2)
    y = Y(:,i);
    newA = [featureMap * y - featureMap * labels; -1];
    newB = - sum(abs(y - labels));
    A(i,:) = newA';
    b(i) = newB';
end


[x, obj, status] = quadprog(H, f, A, b, Aeq, beq, lb, ub, [], options);

if status ~= 1
    fprintf('Warning: quadprog (findWXiPrimal) exited with status %d\n', status);
end

w = x(1:d);
xi = x(d + 1);


% fprintf('Objective %f\n', obj);
