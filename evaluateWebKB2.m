clear;
load webKBResultsFine3


%% get norms

for split = 1:4
    for i = 1:length(Cvec)
        for j = 1:length(kappaVec)
            
            norms(i,j, split) = savedW{split}{i}{j}'*savedW{split}{i}{j};
            
        end
    end
end

%% plot error by norm

figure(1);
subplot(212);
plot(norms(:), trainError(:), '.');
xlabel('||w||^2');
ylabel('training error');
subplot(212);
plot(norms(:), testError(:), '.');
xlabel('||w||^2');
ylabel('testing error');

%% discretize norms

[~, inds] = sort(norms(:));

for i = 1:4
    kappas(:,:,i) = ones(length(Cvec),1) * kappaVec;
end

splits = 5;

cuts = ceil(linspace(0, numel(norms), splits + 1));

fontsize = 12;

for i = 1:splits
    mask = inds(cuts(i)+1:cuts(i+1));
    
    subplot(splits, 2, (i-1) * 2 + 1);
    plot(kappas(mask), trainError(mask), 'x');    
    xlabel('\kappa');
    ylabel('training error');
    title(sprintf('Norms in [%f, %f]', norms(mask(1)), norm(mask(end))), 'FontSize', fontsize);
    set(gca, 'FontSize', fontsize)
    subplot(splits, 2, i * 2);
    plot(kappas(mask), testError(mask), 'x');    
    xlabel('\kappa');
    ylabel('testing error');
    title(sprintf('Norms in [%f, %f]', norms(mask(1)), norm(mask(end))), 'FontSize', fontsize);
    set(gca, 'FontSize', fontsize)
end
