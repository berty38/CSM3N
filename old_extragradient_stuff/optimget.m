function o = optimget(options,name,default,flag)
% Purpose: Obtains the value of an optimization option.
%
% See also: OPTIMSET
%
%% Copyright (c) 1998-2012 MOSEK ApS, Denmark. All rights reserved.

if nargin < 2
    error('Not enough input arguments.');
end
if nargin < 3
    default = [];
end

if ( ~isempty(options) & ~isa(options,'struct') )
    error('First argument must be an options structure created with optimset.');
end

if isempty(options)
    o = default;
    return;
end

%[rcode,onam] = mskoptnam(name);

% MODIFIED by BERT
if isfield(options, name)
    o = getfield(options, name);
else
    o = getfield(default, name);
end
