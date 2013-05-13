function [w, violation, xi] = vanillaM3N(featureMap, labels, scope, S, C, tolerance)

if ~exist('tolerance', 'var')
    tolerance = 1e-6;
end
if ~exist('scope', 'var')
    scope = 1:size(featureMap,2);
end

maxIter = 1000;

d = size(featureMap,1);

options = optimset('algorithm', 'active-set', 'display', 'off');
options.MSK_IPAR_INTPNT_NUM_THREADS = 4;
S.options = options;

violation = inf;
findWXi = @findWXiDual;

%% start learning

w = zeros(d,1);
xi = 0;


Y = inference(w, featureMap, 0, S);

for iter = 1:maxIter
    obj = w'*w / 2 + C * xi;
    
    [y, violation] = sep_oracle(w, labels, featureMap, S, scope);
    Y = [Y y];
    
    subplot(2,3,[1 2]);
    plot(w(1:d - 25));
    title('w');
    subplot(2,3,3);
    imagesc(reshape(w(d-24:d), 5, 5));
    
    subplot(212);
    imagesc(Y);
    title(sprintf('Iteration %d, violation %f, primal objective %f\n', iter, violation - xi, obj));
    drawnow;
    
    if violation - xi < tolerance
        break;
    end
    
    [w, xi, obj] = findWXi(Y, featureMap, labels, C);
    if isempty(w)
        fprintf('Dual has become numerically unstable. Switching to primal\n');
        findWXi = @findWXiPrimal;
        [w, xi, obj] = findWXi(Y, featureMap, labels, C);
    end
end


