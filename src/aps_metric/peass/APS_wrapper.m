function [APS] = APS_wrapper(xi,xn,xo,fs)

options.destDir = './';
options.segmentationFactor = 1; % increase this integer if you experienced "out of memory" problems

% Save inputs as files for the PEASS function:
dir = './';

xi_file = 'tmp_xi.wav';
audiowrite([dir xi_file],xi/max(abs(xi)),fs);
path_in{1} = [dir xi_file];

if ~ isempty(xn)
    xn_file = 'tmp_xn.wav';
    audiowrite([dir xn_file],xn/max(abs(xn)),fs);
    path_in{2} = [dir xn_file];
end

xo_file = 'tmp_xo.wav';
audiowrite([dir xo_file],xo/max(abs(xo)),fs);
path_out = [dir xo_file];

res = PEASS_ObjectiveMeasure(path_in,path_out,options);
APS = res.APS;

system('rm tmp_*.wav');


