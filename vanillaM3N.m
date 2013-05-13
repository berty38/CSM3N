function w = vanillaM3N(featureMap, labels, scope, S, C, tolerance)

if ~exist('scope', 'var')
    scope = 1:size(featureMap,2);
end

options = optimset('display', 'final');
options.MSK_IPAR_INTPNT_NUM_THREADS = 4;
S.options = options;

ell = zeros(size(labels));
ell(scope) = 2*(1-labels(scope)) - 1;

[d, m] = size(featureMap);

conSize = length(S.beq);

f = C*[-featureMap*labels; -S.beq];
H = [speye(d), sparse(d, conSize); sparse(conSize, d+conSize)];

A = [featureMap' S.Aeq'];
b = -ell;

[x, obj, status] = quadprog(H,f,A,b,[],[],[], [], [], options);

w = x(1:d);

