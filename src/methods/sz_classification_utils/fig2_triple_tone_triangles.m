clear all; close all;
save_figures = false; % If true, save the figures to .pdf.


%% Generate the signal:
N = 2^8;
rng(0);
[x, det_zeros, impulses_location] = triple_tone_signal(N);
% [x, ~,~] = triple_impulse_signal_2(N);
det_zeros(det_zeros(:,1)<25 | det_zeros(:,1)>225,:) = [];

% Parameters for the STFT.
Nfft = 2*N;
fmax = 0.5; % Max. norm. frequency to compute the STFT.
[w,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.

% Noise realization:
SNRin = 30;
noise = randn(size(x));
[xnoise,h] = sigmerge(x,noise,SNRin);
gamma =  var(h*noise);
%%
[F,~,~] = tfrstft(x,1:N,Nfft,w,0);
F = F(1:floor(Nfft*fmax),:);
F = flipud(F);
S = abs(F).^2;

figure()
imagesc(S); hold on;
contour(S,[h h],'--r','LineWidth',1.5)
xlim([round(T) N-round(T)])
[F,~,~] = tfrstft(xnoise,1:N,Nfft,w,0);
F = F(1:floor(Nfft*fmax),:);
F = flipud(F);
S = abs(F).^2;
ceros = find_spectrogram_zeros(S);
u=ceros(:,1);
v=ceros(:,2);
plot(v,u,'o','Color','w','MarkerFaceColor','w','MarkerSize',2);
xticks([]); yticks([]);
xticklabels([]); yticklabels([]);
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
% print_figure('figures/level_set_mix_2.pdf',4.0,4.0,'RemoveMargin',true)
%% Check the deterministic zeros and triangles.

[F,~,~] = tfrstft(xnoise,1:N,Nfft,w,0);
F = F(1:floor(Nfft*fmax),:);
F = flipud(F);
S = abs(F).^2;
ceros = find_spectrogram_zeros(S);
u=ceros(:,1);
v=ceros(:,2);
TRI_det = delaunay(det_zeros);

figure()
% imagesc(-log(abs(F))); hold on;
imagesc((abs(F)^2)); hold on;
xlim([round(T) N-round(T)])
% plot([impulses_location(1) impulses_location(1)],[1 N],'--w','LineWidth',0.1);
% plot([impulses_location(2) impulses_location(2)],[1 N],'--w','LineWidth',0.1);
% plot([impulses_location(3) impulses_location(3)],[1 N],'--w','LineWidth',0.1);
triplot(TRI_det,det_zeros(:,1),det_zeros(:,2),'c','LineWidth',0.5);
plot(v,u,'o','Color','w','MarkerFaceColor','w','MarkerSize',2);
% plot([1 N],[det_zeros(4,2) det_zeros(4,2)],'-.w','LineWidth',0.1);
% plot([1 N],[det_zeros(5,2) det_zeros(5,2)],'-.w','LineWidth',0.1);

% plot(det_zeros(:,1),det_zeros(:,2),'ws','MarkerFaceColor','w','MarkerSize',2);
xticks(impulses_location); yticks([]);

xticklabels({'$t_1$','$t_2$','$t_3$'}); yticklabels();% yticklabels({'f_1','f_2'})
xaxisproperties= get(gca, 'XAxis');
xaxisproperties.TickLabelInterpreter = 'latex'; % latex for x-axis
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
% title('Spectrogram and zeros')
% colormap pink
axis square

if save_figures
    print_figure('figures/parallel_tones.pdf',4,4,'RemoveMargin',true)
end

figure()
% imagesc(-log(abs(F))); hold on;
imagesc((abs(F)^2)); hold on;
plot([impulses_location(1) impulses_location(1)],[1 N],'--w','LineWidth',0.1);
plot([impulses_location(2) impulses_location(2)],[1 N],'--w','LineWidth',0.1);
plot([impulses_location(3) impulses_location(3)],[1 N],'--w','LineWidth',0.1);
% plot([1 N],[det_zeros(4,2) det_zeros(4,2)],'-.w','LineWidth',0.1);
% plot([1 N],[det_zeros(5,2) det_zeros(5,2)],'-.w','LineWidth',0.1);
triplot(TRI_det,det_zeros(:,1),det_zeros(:,2),'c','LineWidth',0.5);
plot(det_zeros(:,1),det_zeros(:,2),'ws','MarkerFaceColor','w','MarkerSize',2);
xticks(impulses_location); yticks([]);
xticklabels({'$t_1$','$t_2$','$t_3$'}); %yticklabels({'f_1','f_2'})
xaxisproperties= get(gca, 'XAxis');
xaxisproperties.TickLabelInterpreter = 'latex'; % latex for x-axis
xlabel('time','Interpreter','latex'); % ylabel('frequency')
% title('Spectrogram and zeros')
% colormap pink
xlim([85 171])
% axis square

if save_figures
    print_figure('figures/parallel_tones_det_tri.pdf',1.6779,5,'RemoveMargin',true)
end

%% Find original zeros and triangulation

% Keep zeros within margins:
margin_row = 5; margin_col = 5;
invalid_ceros = zeros(length(ceros),1);
invalid_ceros(ceros(:,1)<margin_row | ceros(:,1)>(size(S,1)-margin_row))=1;
invalid_ceros(ceros(:,2)<margin_col | ceros(:,2)>(size(S,2)-margin_col))=1;
invalid_ceros = logical(invalid_ceros);
valid_ceros = ~invalid_ceros;
% number_of_valid_ceros = sum(valid_ceros);

% ceros = ceros(valid_ceros,:);

% Triangulation of zeros
TRI = delaunay(u,v);
TRI2 =  [];

for j = 1:size(TRI,1)
    if ~any(invalid_ceros(TRI(j,:)))
        TRI2 = [TRI2; TRI(j,:)];
    end
end

[~,MAX_EDGES,TRI_EDGES] = describe_triangles(TRI2,ceros,Nfft,T);

%% Find triangles with an edge larger than certain threshold
LB = 1.3;
longTriangulos=zeros(size(TRI2,1),1);
for i =1:size(TRI2,1)
    if any(TRI_EDGES(i,:)>LB)        
        longTriangulos(i)=1;
    end
end

TRIselected=TRI2(logical(longTriangulos),:);
mask = mask_from_triangles(F,TRIselected,ceros);

% Reconstruction
signal_r = real(sum(F.*mask))/max(w)/N;
QRF = 20*log10(norm(x)/norm(x-signal_r.'));
% plot(x); hold on; plot(signal_r,'r--');

figure()
imagesc(mask)
xlim([round(T) N-round(T)])
figure()
% imagesc(-log(abs(F))); hold on
imagesc((abs(F)^2)); hold on;
xlim([round(T) N-round(T)])
triplot(TRIselected,v,u,'c','LineWidth',0.5);
plot(v,u,'o','Color','w','MarkerFaceColor','w','MarkerSize',2);
xticks([]); yticks([]);
xticklabels([]); yticklabels([]);
xlabel('time','Interpreter','latex'); ylabel('frequency','Interpreter','latex')
title(['$\ell_{max}$='  sprintf('%1.2f',LB)],'Interpreter','latex')
% colormap pink
% axis square
if save_figures
    print_figure('figures/parallel_tones_selected_tri_13.pdf',4.2,4.2,'RemoveMargin',true)
end

