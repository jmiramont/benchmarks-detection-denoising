function F_mat = compForacle_STFT(M,L)

% Compute the sliding tfr window
val = transpose(Fh(-(M/4)+1:((3*M)/4), M, L ));

% Generate a 2D array for later use. Accelerate the computation of the
% cross entropy in 'online_2D'
F_mat = zeros(M,M/2);
F_mat(:,1) = val;
for i = 1:(M/2)-1
    F_mat(:,i+1) = [F_mat(end,i);F_mat(1:end-1,i)];
end
F_mat = F_mat((M/4)+1:((3*M)/4),:); % Truncation to the same lenght than the data
F_mat = F_mat./sum(F_mat);% Normalization