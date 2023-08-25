function [out_signal,fs] = generate_output(filenames)
out_signal = 0;
for i = 1:length(filenames)
    fname = filenames{i};
    [tmp,fs] = audioread(fname);
    out_signal = out_signal+tmp;
end