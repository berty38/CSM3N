function stop = inferenceStat(x,iterationType,i,funEvals,f,t,gtd,g,d,optCond,varargin)

% plot function for inference optimization callback

persistent obj;
persistent prevX;
persistent changeX;
persistent diff;

if i == 0
    obj = [];
    prevX = 0;
    changeX = [];
    diff = [];
end

obj(end+1) = f;
changeX(end+1) = norm(x - prevX);
prevX = x;

interval = 100;
window = 500;

if mod(i-1, interval) == 0
    subplot(411);
    if any(obj <= 0)
        plot(obj);
    else
        semilogy(obj);
    end
    ylabel('objective');
    xlabel('iteration');
    set(gca, 'FontSize', 16);
    subplot(412);
    inds = max(1, length(obj) - window):length(obj);
    plot(inds, obj(inds));
    ylabel('objective');
    xlabel(sprintf('last %d iterations', window));
    title(sprintf('norm of gradient: %d', norm(g)));
    set(gca, 'FontSize', 16);
    
    subplot(413);
    semilogy(changeX);
    ylabel('Change in x');
    xlabel('Iteration');
    
    title(sprintf('Optimality condition %d', optCond));
   
    subplot(414);
    if ~isempty(varargin)
        func = varargin{1};
        
        diff(end+1) = abs(fastDerivativeCheck(func, x));
    end
    plot(diff);
    title('Difference between numerical versus user-supplied derivative');
    
    if diff(end) > 1e10
        keyboard;
    end
    
    drawnow;
end
stop = false;