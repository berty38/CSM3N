clear;
load webKBResultsFine2

%% trim uninteresting C's

filter = 1:length(Cvec);%ismember(Cvec, [0.04, 1, 25]);

Cvec = Cvec(filter);
trainError = trainError(filter, :, :);
testError = testError(filter, :, :);


%%

close all;

font = 'Times New Roman';

avgTr = mean(trainError, 3);
avgTe = mean(testError, 3);

for i = 1:size(avgTr,1)
    smoothedTr(i,:) = smooth(avgTr(i,:), 5);
    smoothedTe(i,:) = smooth(avgTe(i,:), 5);
end

legendStrings = cell(length(Cvec),1);
for i = 1:length(Cvec)
    legendStrings{i} = sprintf('C = %2.2f', Cvec(i));
end

fontsize = 24;
ticksize = 16;
figSize = [0 0 16 9] * 30;

figure('Position', figSize);
clf;

set(gcf,'PaperPositionMode','auto')
plot(kappaVec(1:size(avgTr,2)), avgTr, 'x', 'LineWidth', 1, 'MarkerSize', 10);
hold on;
plot(kappaVec(1:size(avgTr,2)), smoothedTr, '-', 'LineWidth', 2);
hold off;

ax = axis;
axis([-.1, kappaVec(end)+.1, ax(3)-.01, ax(4)+.02]);

ylabel('Training error', 'FontSize', fontsize, 'FontName', font);
xlabel('\kappa', 'FontSize', fontsize, 'FontName', font);

set(gca, 'FontSize', ticksize, 'FontName', font, 'ygrid', 'on', 'TickDir', 'out', 'Box', 'off')
legend(legendStrings, 'Location', 'NorthWest', 'FontSize', 12)

print('-depsc', 'train.eps');

figure('Position', figSize);
clf;

set(gcf,'PaperPositionMode','auto')
plot(kappaVec(1:size(avgTe,2)), avgTe, 'x', 'LineWidth', 1, 'MarkerSize', 10);
hold on;
plot(kappaVec(1:size(avgTr,2)), smoothedTe, '-', 'LineWidth', 2);
hold off;

ax = axis;
axis([-.1, kappaVec(end)+.1, ax(3)-.01, ax(4)+.02]);

ylabel('Testing error', 'FontSize', fontsize, 'FontName', font);
xlabel('\kappa', 'FontSize', 24, 'FontName', font);
set(gca, 'FontSize', ticksize, 'FontName', font, 'ygrid', 'on', 'TickDir', 'out', 'Box', 'off')
print('-depsc', 'test.eps');


figure('Position', figSize);
clf;

set(gcf,'PaperPositionMode','auto')
plot(kappaVec(1:size(avgTe,2)), avgTe - avgTr, 'x', 'LineWidth', 1, 'MarkerSize', 10);
hold on;
plot(kappaVec(1:size(avgTr,2)), smoothedTe - smoothedTr, '-', 'LineWidth', 2);
hold off;

ax = axis;
axis([-.1, kappaVec(end)+.1, ax(3)-.01, ax(4)+.02]);

ylabel('Pseudo-defect', 'FontSize', fontsize, 'FontName', font);
xlabel('\kappa', 'FontSize', 24, 'FontName', font);
set(gca, 'FontSize', ticksize, 'FontName', font, 'ygrid', 'on', 'TickDir', 'out', 'Box', 'off')
print('-depsc', 'defect.eps');
