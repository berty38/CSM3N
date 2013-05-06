function [w, violation, xi] = stableSVMStruct(featureMap, labels, S, C, kappa, tolerance, scope, baseD, k)

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

violation = inf;
findWXi = @findWXiDual;

%% start learning

w = zeros(d,1);
xi = 0;


Y = inference(w, 1.0, featureMap, S);

for iter = 1:maxIter
    obj = w'*w / (kappa * 2) + C * xi;
    
    [y, violation] = sep_oracle(w, kappa, labels, featureMap, S, scope);
    Y = [Y y];
    
    if violation - xi < tolerance
        break;
    end
    
    subplot(212);
    imagesc(Y);
    title(sprintf('Iteration %d, violation %f, primal objective %f\n', iter, violation - xi, obj));
    drawnow;
    
    if nargin == 9
        subplot(221);
        plot(w(1:baseD*k), 'x');
        axis([0 baseD*k+1 min(w)-0.01 max(w)+0.01]);
        title('Local weights');
        subplot(222);
        imagesc(reshape(w(baseD*k+1:end), k, k));
        colorbar;
        title('Edge weights');
    else
        subplot(211);
        plot(w, 'x');
    end
    
    %     while true
    [w, xi, obj] = findWXi(Y, kappa, featureMap, labels, C);
    if isempty(w)
        fprintf('Dual has become numerically unstable. Switching to primal\n');
        findWXi = @findWXiPrimal;
        [w, xi, obj] = findWXi(Y, kappa, featureMap, labels, C);
    end
    %
    %         % find kappa
    %         oldKappa = kappa;
    %         kappa = findKappa(Y, featureMap, labels, w, xi);
    %         fprintf('kappa %f, obj %f\n', kappa, obj);
    %         if abs(kappa - oldKappa) < 1e-5
    %             break;
    %         else
    %             keyboard;
    %         end
    %     end
end


