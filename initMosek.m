global mosek_path
if isempty(mosek_path)
    mosek_path = '/Users/bert/Dropbox/Research/tools/mosek/6';
end

dyld_path = sprintf('%s/tools/platform/osx64x86/bin', mosek_path);

oldpath = getenv('DYLD_LIBRARY_PATH');

setenv('DYLD_LIBRARY_PATH', sprintf('%s:%s', dyld_path, oldpath));
%setenv('DYLD_LIBRARY_PATH', dyld_path);

setenv('MOSEKLM_LICENSE_FILE', ...
    sprintf('%s/licenses/mosek.lic', mosek_path));

addpath(sprintf('%s/tools/platform/osx64x86/bin', mosek_path));
addpath(sprintf('%s/toolbox/r2012a', mosek_path));

currdir = pwd;

cd(sprintf('%s/tools/platform/osx64x86/bin', mosek_path));
mosekopt;
cd(currdir);

clear dyld_path oldpath currdir;
