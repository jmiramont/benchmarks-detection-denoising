% Study of the performance of a denosing strategy, based on an unsupervised
% method to classify the zeros of the spectrogram. The first section of
% the script runs the experiment. Alternatively, you can run only the
% second section, provided that the file with the results is in the same
% folder or in the path.
%
% September 2022
% Author: Juan M. Miramont <juan.miramont@univ-nantes.fr>
% -------------------------------------------------------------------------

%% Section 1. Simulations:
clear all;
% Signal Length:
N = 2^10;

folder = './';
%% required paths
folder = './';
addpath(folder);
% addpath(strcat([folder 'PB_method']));
addpath(genpath([folder 'PB_method']));
addpath(genpath([folder 'contour_filtering_utils']));
% addpath(strcat([folder 'tools']));
% addpath(strcat([folder 'synchrosqueezedSTFT']));
% addpath(strcat([folder 'PseudoBay']));

%%
% Parameters for the STFT.
Nfft = 2*N;
fmax = 0.5; % Max. norm. frequency to compute the STFT.
[w,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.

% Generate a signal with three parallel tones.
% [x, det_zeros, impulses_location] = triple_tone_signal(N);

load McSyntheticMixture.mat
% load McMultiLinear2.mat
x = x.';
% Compute STFT and spectrogram.
xs = [x(N/2+1:-1:2); x; -x(N-1:-1:N/2)];
[F,~,~] = tfrstft(xs,N/2+1:N/2+N,Nfft,w,0);
S = abs(F(1:Nfft/2+1,:).^2);

% Noise realizations:
J = 256;
K = 150;

% Save the noise realizations for later.
rng(0);
noise_matrix = randn(K,N);
noise_matrix = noise_matrix./std(noise_matrix,[],2);

% Triangles with an edge longer than lmax are selected, according to the
% algorithm proposed by P. Flandrin in " Time-frequency filtering based on
% spectrogram zeros", IEEE Signal Procesing Letters, 2015.
% lmax = 1.3:0.1:1.6;

% Simulate different SNRs.
SNRs = -20:10:20;


%%
for q = 1:length(SNRs)
    disp(q);
    SNRin = SNRs(q);
    for k = 1:K
        noise = noise_matrix(k,:);
        xnoise = sigmerge(x,noise.',SNRin);
        Tind = round(2*T)+1;

        % Filtering using classification zeros:
        
%         tic();
%         [mask_na, signal_r, TRI, TRIselected, ceros, F, class, nclust(q,k)] =...
%             classified_zeros_denoising(xnoise, 'estimate', J, {'gmm','gap'} , [2,2], 1.0);
%         tiempos(q,k,1) = toc();
% 
%         QRF(q,k,1) = 20*log10(norm(x(Tind:end-Tind))/...
%             norm(x(Tind:end-Tind)-signal_r(Tind:end-Tind)));
% 
%         MSE(q,k,1) = my_mse(x(Tind:end-Tind),signal_r(Tind:end-Tind));
%         CCC(q,k,1) = my_corcoef(x(Tind:end-Tind),signal_r(Tind:end-Tind));
% 
%         % Filtering using classification zeros:
%         
%         tic();
%         [mask_na, signal_r, TRI, TRIselected, ceros, F, class, nclust(q,k)] =...
%             classified_zeros_denoising(xnoise, 'estimate', J, {'gmm','ch'} , [2,2], 1.0);
%         tiempos(q,k,2) = toc();
% 
%         QRF(q,k,2) = 20*log10(norm(x(Tind:end-Tind))/...
%             norm(x(Tind:end-Tind)-signal_r(Tind:end-Tind)));
% 
%         MSE(q,k,2) = my_mse(x(Tind:end-Tind),signal_r(Tind:end-Tind));
%         CCC(q,k,2) = my_corcoef(x(Tind:end-Tind),signal_r(Tind:end-Tind));
% 
%         % Filtering using classification zeros:
%         
%         tic();
%         [mask_na, signal_r, TRI, TRIselected, ceros, F, class, nclust(q,k)] =...
%             classified_zeros_denoising(xnoise, 'estimate', J, {'kmeans','gap'} , [2,2], 1.0);
%         tiempos(q,k,3) = toc();
% 
%         QRF(q,k,3) = 20*log10(norm(x(Tind:end-Tind))/...
%             norm(x(Tind:end-Tind)-signal_r(Tind:end-Tind)));
%         
%         MSE(q,k,3) = my_mse(x(Tind:end-Tind),signal_r(Tind:end-Tind));
%         CCC(q,k,3) = my_corcoef(x(Tind:end-Tind),signal_r(Tind:end-Tind));
% 
%         % Filtering using classification zeros:
%         tic();
%         [mask_na, signal_r, TRI, TRIselected, ceros, F, class, nclust(q,k)] =...
%             classified_zeros_denoising(xnoise, 'estimate', J, {'kmeans','ch'} , [2,2], 1.0);
%         tiempos(q,k,4) = toc();
% 
%         QRF(q,k,4) = 20*log10(norm(x(Tind:end-Tind))/...
%             norm(x(Tind:end-Tind)-signal_r(Tind:end-Tind)));
% 
%         MSE(q,k,4) = my_mse(x(Tind:end-Tind),signal_r(Tind:end-Tind));
%         CCC(q,k,4) = my_corcoef(x(Tind:end-Tind),signal_r(Tind:end-Tind));
%         
%         % Filtering using DT method:
%         tic();    
%         signal_r = dt_method(xnoise, 1.2);
%         tiempos(q,k,5) = toc();
% 
%         QRF(q,k,5) = 20*log10(norm(x(Tind:end-Tind))/...
%             norm(x(Tind:end-Tind)-signal_r(Tind:end-Tind).'));
% 
%         MSE(q,k,5) = my_mse(x(Tind:end-Tind),signal_r(Tind:end-Tind));
%         CCC(q,k,5) = my_corcoef(x(Tind:end-Tind),signal_r(Tind:end-Tind));
% 
%         % Filtering using DT method:
%         tic();
%         signal_r = dt_method(xnoise, 1.5);
%         tiempos(q,k,6) = toc();
% 
%         QRF(q,k,6) = 20*log10(norm(x(Tind:end-Tind))/...
%             norm(x(Tind:end-Tind)-signal_r(Tind:end-Tind).'));
%         
%         MSE(q,k,6) = my_mse(x(Tind:end-Tind),signal_r(Tind:end-Tind));
%         CCC(q,k,6) = my_corcoef(x(Tind:end-Tind),signal_r(Tind:end-Tind));
%         % Filtering using Brevdo method:
%         tic()
%         signal_r = brevdo_method(xnoise, 2*3, true);
%         tiempos(q,k,7) = toc();
% 
%         QRF(q,k,7) = 20*log10(norm(x(Tind:end-Tind))/...
%             norm(x(Tind:end-Tind)-signal_r(Tind:end-Tind).'));
%         
%         MSE(q,k,7) = my_mse(x(Tind:end-Tind),signal_r(Tind:end-Tind));
%         CCC(q,k,7) = my_corcoef(x(Tind:end-Tind),signal_r(Tind:end-Tind));
%         
%         % Filtering with HT
%         tic()
%         [signal_r, mask] = hard_thresholding(xnoise,2.5);
%         tiempos(q,k,8) = toc();
% 
%         QRF(q,k,8) = 20*log10(norm(x(Tind:end-Tind))/...
%             norm(x(Tind:end-Tind)-signal_r(Tind:end-Tind).'));
%         
%         MSE(q,k,8) = my_mse(x(Tind:end-Tind),signal_r(Tind:end-Tind));
%         CCC(q,k,8) = my_corcoef(x(Tind:end-Tind),signal_r(Tind:end-Tind));
%         
%         tic()
%         [signal_r, mask] = hard_thresholding(xnoise,3.0);
%         tiempos(q,k,9) = toc();
% 
%         QRF(q,k,9) = 20*log10(norm(x(Tind:end-Tind))/...
%             norm(x(Tind:end-Tind)-signal_r(Tind:end-Tind).'));
%         
%         MSE(q,k,9) = my_mse(x(Tind:end-Tind),signal_r(Tind:end-Tind));
%         CCC(q,k,9) = my_corcoef(x(Tind:end-Tind),signal_r(Tind:end-Tind));

%         % Filtering using Contours:
        tic();
        signal_r = contour_filtering(xnoise, 3,[],[],N/2);
        tiempos(q,k,10) = toc();

        QRF(q,k,10) = 20*log10(norm(x(Tind:end-Tind))/...
            norm(x(Tind:end-Tind)-signal_r(Tind:end-Tind).'));
        
        MSE(q,k,10) = my_mse(x(Tind:end-Tind),signal_r(Tind:end-Tind));
        CCC(q,k,10) = my_corcoef(x(Tind:end-Tind),signal_r(Tind:end-Tind));
%         
% %         % Filtering using Contours:
%         tic();
%         signal_r = contour_filtering(xnoise, 3*2,[],[],N/2);
%         tiempos(q,k,8) = toc();
% 
%         QRF(q,k,8) = 20*log10(norm(x(Tind:end-Tind))/...
%             norm(x(Tind:end-Tind)-signal_r(Tind:end-Tind).'));        
%         
% %         Filtering using PB method:
%         tic();
%         signal_r = pb_method(xnoise, 3, true);
%         tiempos(q,k,9) = toc();
% 
%         QRF(q,k,9) = 20*log10(norm(x(Tind:end-Tind))/...
%             norm(x(Tind:end-Tind)-signal_r(Tind:end-Tind).'));
% 
% %          Filtering using PB method:
%         tic();
%         signal_r = pb_method(xnoise, 3*2, true);
%         tiempos(q,k,10) = toc();
% 
%         QRF(q,k,10) = 20*log10(norm(x(Tind:end-Tind))/...
%             norm(x(Tind:end-Tind)-signal_r(Tind:end-Tind).'));
        
            
    end
end

note = "The K selection was between 2 and 3 (not 1!).";
% Save results.
save variables_benchmark_AREA_RATIO_MAX_KMEANS_FALSE_NONORMALIZE_FULL_J256_non.mat

%% Section 2: Load the results and generate the figures.
clear all;
% load variables_benchmark_AREA_RATIO_ENTROPY_CITYBLOCK.mat
% load variables_benchmark_AREA_RATIO_MAX_GMM_FALSE_NONORMALIZE_B100_J256.mat
% load variables_benchmark_AREA_RATIO_MAX_GMM_FALSE_NONORMALIZE_B100_J256_more_methods.mat
% load variables_benchmark_AREA_RATIO_MAX_KMEANS_FALSE_NONORMALIZE_B100_J256_more_methods.mat

load variables_benchmark_AREA_RATIO_MAX_KMEANS_FALSE_NONORMALIZE_FULL_J256_non.mat


% load variables_benchmark_AREA_RATIO_MAX_GMM_FALSE_NONORMALIZE_KH_J256.mat

% load variables_benchmark_6_JUST_AREA_RATIO.mat
% load variables_benchmark_6_JUST_AREA_RATIO_BETA_085.mat
% load variables_benchmark_6_JUST_AREA_RATIO_BETA_1_CALINSKY.mat
% load variables_benchmark_6_AREA_RATIO_BARYDIST_NONORMALIZATION.mat
% load variables_benchmark_AREA_RATIO_FARP_CITYBLOCK.mat

colors = {'#0072BD','#D95319','#EDB120','#7E2F8E','#77AC30','#4DBEEE','#A2142F'};
markers = {'o','x','d','s','pentagram','^','p','v','^'};

method_name = string({'SZC-GMM-GAP','SZC-GMM-CaHa','SZC-KMEANS-GAP','SZC-KMEANS-CaHa','DT-1.2','DT-1.5','SST+RD','HT - 2.5','HT','Contours'});
% markers = string({'-o','-v','-^','-d','-s'});
figure();
mark_count = 0;
for i = [1,5,6,7,9,10] %1:size(QRF,3)
    qrf = mean(QRF(:,:,i).');
    % plot(SNRs,qrf,'DisplayName',method_name(i)); hold on
    errorbar(SNRs,...
        qrf,...
        std(QRF(:,:,i).'),...
        'CapSize',0.6,...
        'DisplayName',method_name(i),...
        'Marker',markers{mod(mark_count,9)+1},...
        'MarkerSize',4.0,...
        'Color',colors{mod(i,7)+1});
%         'MarkerFaceColor',colors{mod(i,7)+1})
%         );
mark_count = mark_count +1;
hold on;

end
xlabel('SNRin (dB)','Interpreter','latex'); 
ylabel('QRF (dB)','Interpreter','latex');
ylim([-20, 33])
xlim([-21, 21])
grid on;

legend('Location','northwest');
legend('boxoff')
% axes('Position',[.6 .1150 .005 .005]);
% box off
%
% imagesc(flipud(S));
% xticklabels([]); yticklabels([]);xticks([]); yticks([]);

% savefig('snr_bench.fig')
% print_figure('snr_bench.pdf',6.0,8.0,'RemoveMargin',true)

%%
figure();
for i = [1,2,3,4] %1:size(QRF,3)
    qrf = mean(QRF(:,:,i).');
    % plot(SNRs,qrf,'DisplayName',method_name(i)); hold on
    errorbar(SNRs,...
        qrf,...
        2*std(QRF(:,:,i).')/sqrt(K),...
        'CapSize',0.6,...
        'DisplayName',method_name(i),...
        'Marker',markers{mod(i,9)+1},...
        'MarkerSize',4.0);
%         'Color',colors{mod(i,7)+1});
%         'MarkerFaceColor',colors{mod(i,7)+1})
%         );

hold on;

end
xlabel('SNRin (dB)','Interpreter','latex'); 
ylabel('QRF (dB)','Interpreter','latex');
ylim([-20, 33])
xlim([-21, 21])
grid on;

legend('Location','northwest');
legend('boxoff')

% savefig('qrf_bench_only_ours.fig')
% print_figure('qrf_bench_only_ours.pdf',6.0,8.0,'RemoveMargin',true)

%%
figure();
mark_count = 0;
for i = [1,5,6,7,9,10] %1:size(QRF,3)
    qrf = mean(CCC(:,:,i).');
    % plot(SNRs,qrf,'DisplayName',method_name(i)); hold on
    errorbar(SNRs,...
        qrf,...
        std(CCC(:,:,i).'),...
        'CapSize',0.6,...
        'DisplayName',method_name(i),...
        'Marker',markers{mod(mark_count,9)+1},...
        'MarkerSize',4.0,...
        'Color',colors{mod(i,7)+1});
%         'MarkerFaceColor',colors{mod(i,7)+1})
%         );

hold on;
mark_count = mark_count +1;
end
xlabel('SNRin (dB)','Interpreter','latex'); 
ylabel('CC','Interpreter','latex');
ylim([0, 1.3])
xlim([-21, 21])
grid on;

legend('Location','northwest');
legend('boxoff')
% axes('Position',[.6 .1150 .005 .005]);
% box off
%
% imagesc(flipud(S));
% xticklabels([]); yticklabels([]);xticks([]); yticks([]);

% savefig('ccc_bench.fig')
% print_figure('ccc_bench.pdf',6.0,8.0,'RemoveMargin',true)



%% Execution time
clc
for i = 1:size(tiempos,3)
    ttt = tiempos(:,:,i);
    mean(ttt(:))
end
