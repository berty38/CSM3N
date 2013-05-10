% add necessary paths for using minFunc and minConf

% This is not the ideal solution to add minFunc and minConf to the path,
% but for some weird reason I can't edit my path.
% NOTE: These packages must be installed already.

% Change this path to the parent directory of wherever you keep these
% packages.
pathToPackages = '/Users/blondon/Code/MATLAB/';

% install minFunc 2012
addpath([pathToPackages 'minFunc_2012/'])
addpath([pathToPackages 'minFunc_2012/minFunc/'])
addpath([pathToPackages 'minFunc_2012/minFunc/compiled/'])
addpath([pathToPackages 'minFunc_2012/autoDif/'])

% install minConf
addpath([pathToPackages 'minConf/'])
addpath([pathToPackages 'minConf/minConf'])
addpath([pathToPackages 'minConf/minFunc'])
