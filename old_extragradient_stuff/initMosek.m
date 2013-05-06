global mosek_path
if isempty(mosek_path)
    mosek_path = '/Users/bert/Dropbox/Research/tools/mosek';
end

dyld_path = sprintf('%s/6/tools/platform/osx64x86/bin', mosek_path);

oldpath = getenv('DYLD_LIBRARY_PATH');

setenv('DYLD_LIBRARY_PATH', sprintf('%s:%s', dyld_path, oldpath));
%setenv('DYLD_LIBRARY_PATH', dyld_path);

setenv('MOSEKLM_LICENSE_FILE', ...
    sprintf('%s/6/licenses/mosek.lic', mosek_path));

addpath(sprintf('%s/6/tools/platform/osx64x86/bin', mosek_path));
addpath(sprintf('%s/6/toolbox/r2009b', mosek_path));

currdir = pwd;

cd(sprintf('%s/6/tools/platform/osx64x86/bin', mosek_path));
mosekopt;
cd(currdir);

clear dyld_path oldpath currdir;
