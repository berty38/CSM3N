function [w, kappa, y] = dualExtragradientLearn(featureMap, labels, C, S, scope, nu)

[d, m] = size(featureMap);

w = zeros(d,1);
kappa = 0;
y = labels;

ell = zeros(m,1);
ell(scope) = 1 - labels(scope);

converged = false;

eta = 0.1;

tolerance = 1e-6;

iter = 1;

sw = 0;
sk = 0;
sy = 0;

while ~converged
    oldW = w;
    oldKappa = kappa;
    oldY = y;
    
    % compute prediction step
    wP = w - eta * sw;
    kappaP = max(0, kappa - eta * sk);
    yP = euclideanProject(S, y + eta * sy);
    
    [gradWc, gradKappac, gradYc] = fullGradient(wP, kappaP, C, featureMap, labels, yP, ell);
    
    % update variables
    w = w - eta * gradWc;
    kappa = max(0, kappa - eta * gradKappac);
    y = euclideanProject(S, y + eta * gradYc);
    
    [gradW, gradKappa, gradY] = fullGradient(w, kappa, C, featureMap, labels, y, ell);
    
    sw = sw - gradW;
    sk = sk - gradKappa;
    sy = sy + gradY;
    
    change = norm([oldW; oldKappa; oldY] - [w; kappa; y]);
    
    if change < tolerance
        converged = true;
    end
    
    % compute current objective and error
    obj(iter) = 0.5 * (w'*w) / (C * (kappa + 1)) + w' * featureMap * y - 0.5 * kappa * (y'*y) + ell' * y - ...
        w' * featureMap * labels + nnz(labels) - 0.5 * kappa * (labels'*labels);
    
    prediction = inference(w, kappa, featureMap, S);
    err(iter) = mean(abs(prediction(scope) - labels(scope)));
    
    if mod(iter - 1, 20) == 0
        subplot(411);
        plot(y, 'x');
        title('y');
        subplot(412);
        plot(w, 'x');
        title('w');
        subplot(413);
        plot(obj);
        title(sprintf('Objective. kappa = %f', kappa));
        subplot(414);
        plot(err);
        title('training error');
        drawnow;
        
        fprintf('Iteration %d, change %f\n', iter, change);
    end
    
    iter = iter + 1;
end


function [gradW, gradKappa, gradY] = fullGradient(w, kappa, C, featureMap, labels, y, ell)

gradW = w / (C * (kappa + 1)) + featureMap * (y - labels);
gradKappa = (labels'*labels - y'*y) / 2 - w'*w / (2 * C * (kappa + 1)^2);
gradY = featureMap' * w - kappa * y + ell;
