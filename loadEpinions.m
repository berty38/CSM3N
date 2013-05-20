%% Load data
epinions = load('epinions/epinions-small.txt');
n = max(max(epinions(:,1:2)));
gt = sparse(epinions(:,1),epinions(:,2),2*epinions(:,3)-1,n,n);
inconsistent = nnz(abs(triu(gt-gt')) == 2)
unrecip_pos = nnz(triu(gt==1 & gt'==0)) + nnz(triu(gt==0 & gt'==1))
unrecip_neg = nnz(triu(gt==-1 & gt'==0)) + nnz(triu(gt==0 & gt'==-1))
recip_pos = nnz(triu(gt==gt' & gt==1))
recip_neg = nnz(triu(gt==gt' & gt==-1))

% load 'epinions/epinions-big.txt'
% epinions_big(:,1:2) = epinions_big(:,1:2) + 1;
% gt = spconvert(epinions_big);

%% Snowball sample a reasonably sized connected component

