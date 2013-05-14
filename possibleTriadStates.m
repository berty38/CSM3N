function s = possibleTriadStates(v1, v2, v3)

% Returns array of possible states that a partially observed triad 
% (v1,v2,v3) can be in, where v1,v2,v3 in {-1,0,+1}

if v1 == 0
	s1 = [0;1];
else
	s1 = (v1 + 1) / 2;
end

if v2 == 0
	s2 = [0;1];
else
	s2 = (v2 + 1) / 2;
end

if v3 == 0
	s3 = [0;1];
else
	s3 = (v3 + 1) / 2;
end

s = bsxfun(@plus, s2, s1'*2);
s = bsxfun(@plus, s3, s(:)'*2);
s = s(:) + 1;
