function [train, test] = snowballSample(graph)

% Get all edges
[I,J] = find(graph);
n_e = length(I);

front_tr = zeros(n_e,2);
front_te = zeros(n_e,2);
next_front_tr = 1;
next_front_te = 1;

%% Randomly sample train/test seeds
seed = randsample(n_e, 2);
% train
i = I(seed(1)); j = J(seed(1));
graph(i,j) = 0;
train = [i j];
front_tr(next_front_tr,:) = [i j];
n_e = n_e - 1;
next_front_tr = next_front_tr + 1;
% test
i = I(seed(2)); j = J(seed(2));
graph(i,j) = 0;
test = [i j];
front_te(next_front_te,:) = [i j];
n_e = n_e - 1;
next_front_te = next_front_te + 1;

%% Grab all edges until none left
while n_e > 0
	%% Train
	% Try to find edges adjacent to frontier
	added = 0;
	while added == 0 && nnz(front_tr(:,1)) > 0
		% Randomly choose edge on frontier
		frontIdx = find(front_tr(:,1));
		if length(frontIdx) > 1
			idx = randsample(frontIdx, 1);
		else
			idx = frontIdx;
		end
		i = front_tr(idx,1); j = front_tr(idx,2);
		% Update frontier
		front_tr(idx,:) = [0 0];
		neighbors_i = find(graph(i,:))';
		neighbors_j = find(graph(j,:))';
		% If any adjacent edges
		if length(neighbors_i) + length(neighbors_j) > 0
			edges = [repmat(i,length(neighbors_i),1) neighbors_i; ...
					 repmat(j,length(neighbors_j),1) neighbors_j];
			added = size(edges,1);
			% Add to set/frontier and update graph/remaining edge count
			graph(edges(:,1),edges(:,2)) = 0;
			train = [train ; edges];
			front_tr(next_front_tr:next_front_tr+added-1,:) = edges;
			n_e = n_e - added;
			next_front_tr = next_front_tr + added;
		end
	end
	% If couldn't find any edges adjacent to frontier, reseed
	if added == 0
		[I,J] = find(graph);
		seed = randsample(n_e, 1);
		i = I(seed); j = J(seed);
		graph(i,j) = 0;
		train = [train ; i j];
		front_tr(next_front_tr,:) = [i j];
		n_e = n_e - 1;
		next_front_tr = next_front_tr + 1;
	end
	
	% Check for remaining edges
	if n_e == 0
		break
	end
	
	%% Test
	% Try to find edges adjacent to frontier
	added = 0;
	while added == 0 && nnz(front_te(:,1)) > 0
		% Randomly choose edge on frontier
		frontIdx = find(front_te(:,1));
		if length(frontIdx) > 1
			idx = randsample(frontIdx, 1);
		else
			idx = frontIdx;
		end
		i = front_te(idx,1); j = front_te(idx,2);
		% Update frontier
		front_te(idx,:) = [0 0];
		neighbors_i = find(graph(i,:))';
		neighbors_j = find(graph(j,:))';
		% If any adjacent edges
		if length(neighbors_i) + length(neighbors_j) > 0
			edges = [repmat(i,length(neighbors_i),1) neighbors_i; ...
					 repmat(j,length(neighbors_j),1) neighbors_j];
			added = size(edges,1);
			% Add to set/frontier and update graph/remaining edge count
			graph(edges(:,1),edges(:,2)) = 0;
			test = [test; edges];
			front_te(next_front_te:next_front_te+added-1,:) = edges;
			n_e = n_e - added;
			next_front_te = next_front_te + added;
		end
	end
	% If couldn't find any edges adjacent to frontier, reseed
	if added == 0
		[I,J] = find(graph);
		seed = randsample(n_e, 1);
		i = I(seed); j = J(seed);
		graph(i,j) = 0;
		test = [test ; i j];
		front_te(next_front_te,:) = [i j];
		n_e = n_e - 1;
		next_front_te = next_front_te + 1;
	end
end

%% Convert edge lists to graphs
n = size(graph,1);
train = sparse(train(:,1), train(:,2), ones(size(train,1),1), n, n);
test = sparse(test(:,1), test(:,2), ones(size(test,1),1), n, n);



