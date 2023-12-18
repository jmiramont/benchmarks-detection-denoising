function [zeros_pos, Qz] = find_spectrogram_zeros(S)
% Find zeros as minima of the spectrogram S in a local 3x3 grid.
%
% Input:
% - S: The spectrogram of a signal.
%
% Output:
% - zeros_pos:  A [N,2] array with the time-frequency coordenates of the
%               zeros of the spectrogram. Where N is the number of zeros.
% - Qz:         A matrix with the same dimension as S, with 1 if there is a
%               zero in a given t-f position, or 0 otherwise.
%
% September 2022
% Author: Juan M. Miramont <juan.miramont@univ-nantes.fr>
% -------------------------------------------------------------------------


N1 = size(S,1);
M1 = size(S,2);
S=[Inf*ones(N1,1) S Inf*ones(N1,1)];
S=[Inf*ones(1,M1+2); S; Inf*ones(1,M1+2)];

Qaux = S<circshift(S,1,1) & S<circshift(S,-1,1) & ... % Up and down
     S<circshift(S,1,2) & S<circshift(S,-1,2) & ... % Left and right
     S<circshift(S,[1,1]) & S<circshift(S,[-1,-1])  & ... % Diagonals
     S<circshift(S,[-1,1]) & S<circshift(S,[1,-1]);

Qz = Qaux(2:end-1,2:end-1);
[row,col] = ind2sub(size(Qz),find(Qz));
zeros_pos = [row,col];