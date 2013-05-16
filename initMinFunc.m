% add necessary paths for using minFunc and minConf

% This is not the ideal solution to add minFunc and minConf to the path,
% but for some weird reason I can't edit my path.
% NOTE: These packages must be installed already.

% Change this path to the parent directory of wherever you keep these
% packages.
global minFunc_path
if isempty(minFunc_path)
	minFunc_path = '~/Dropbox/Research/tools/';
end

% install minFunc 2012
addpath([minFunc_path 'minFunc_2012/'])
addpath([minFunc_path 'minFunc_2012/minFunc/'])
addpath([minFunc_path 'minFunc_2012/minFunc/compiled/'])
addpath([minFunc_path 'minFunc_2012/autoDif/'])

% install minConf
addpath([minFunc_path 'minConf/'])
addpath([minFunc_path 'minConf/minConf'])
addpath([minFunc_path 'minConf/minFunc'])
