clear;
%  file1 = 'webKBFullNoLinksResultsJoint3Ex';
%  file3 = 'webKBFullNoLinksResultsJoint3Ex';
 file1 = 'webKBFullResultsJoint1Ex';
 file3 = 'webKBFullResultsJoint3Ex';
% file1 = 'webKBSynthResultsJoint1Ex';
% file3 = 'webKBSynthResultsJoint3Ex';
% file1 = 'webKBSmallResultsJoint1Ex';
% file3 = 'webKBSmallResultsJoint3Ex';
%%
load(file1);

set(0,'DefaultAxesFontSize',14)

figure(1);

clf;
subplot(211);
semilogx(Cvec, mean(testError(:,:,1), 2), 'r');
hold on;
semilogx(Cvec, mean(trainError(:,:,1), 2), 'r--');
semilogx(Cvec, mean(testError(:,:,2), 2), 'b');
semilogx(Cvec, mean(trainError(:,:,2), 2), 'b--');
hold off;
legend('M3N-test', 'M3N-train', 'CSM3N-test', 'CSM3N-train', 'Location', 'Best');
title('Train on 1, test on 3');
xlabel('C');
ylabel('avg. per-page error');

load(file3);

subplot(212);
semilogx(Cvec, mean(testError(:,:,1), 2), 'r');
hold on;
semilogx(Cvec, mean(trainError(:,:,1), 2), 'r--');
semilogx(Cvec, mean(testError(:,:,2), 2), 'b');
semilogx(Cvec, mean(trainError(:,:,2), 2), 'b--');
hold off;
legend('M3N-test', 'M3N-train', 'CSM3N-test', 'CSM3N-train', 'Location', 'Best');
title('Train on 3, test on 1');
xlabel('C');
ylabel('avg. per-page error');

%%
figure(2);
clf;
load(file1);
for i = 1:length(schools)
    subplot(2,4,i);
    semilogx(Cvec, testError(:,i,1), 'r-x');
    hold on;
    semilogx(Cvec, trainError(:,i,1), 'r--x');
    semilogx(Cvec, testError(:,i,2), 'b-x');
    semilogx(Cvec, trainError(:,i,2), 'b--x');
    hold off;
    title(sprintf('Train on %s, test on rest', schools{i}));
    xlabel('C');
    ylabel('per-page error');
end

load(file3);
for i = 1:length(schools)
    subplot(2,4,4+i);
    semilogx(Cvec, testError(:,i,1), 'r-x');
    hold on;
    semilogx(Cvec, trainError(:,i,1), 'r--x');
    semilogx(Cvec, testError(:,i,2), 'b-x');
    semilogx(Cvec, trainError(:,i,2), 'b--x');
    hold off;
    title(sprintf('Train on 3, test on %s', schools{i}));
    xlabel('C');
    ylabel('per-page error');
end
legend('M3N-test', 'M3N-train', 'CSM3N-test', 'CSM3N-train', 'Location', 'SouthEast');
