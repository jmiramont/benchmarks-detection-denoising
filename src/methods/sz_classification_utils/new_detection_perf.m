% Study of the detection performance. The comparison with the detection
% tests used in Bardenet, Flamant and Chainais (2018) is done in python in
% a separate folder.
%
%--------------------------------------------------------------------------

close all;
clear all;

Ns = [2^7 2^8 2^9];
Js = round(logspace(7*log10(2),10*log10(2),10));
reps = 50;

%%
for k = 1:length(Ns)

    N = Ns(k);
    % Parameters for the STFT.
    Nfft = 2*N;
    [~,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.
    dT = 3*ceil(T/8);

    % Generate the signal.
    Nchirp = N;
    tmin = round((N-Nchirp)/2);
    tmax = tmin + Nchirp;
    x = zeros(N,1);
    instf1 = 0.1+0.3*(0:Nchirp-1)/Nchirp;
    x(tmin+1:tmax) = (cos(2*pi*cumsum(instf1))).*tukeywin(Nchirp,0.5).';

    % STFT and Spectrogram
    [w,T] = roundgauss(Nfft,1e-6); % Round Gaussian window.
    [F,~,~] = tfrstft(x,1:N,Nfft,w,0);
    F = F(1:N+1,:);
    F = flipud(F);
    S = abs(F).^2;
    % figure(); imagesc(S);


    rng(0)
    noise_matrix = randn(reps,N)+1i*randn(reps,N);
    % save('noise_matrix_64.mat','noise_matrix','x');

    for j = 1:length(Js)
        J = Js(j);
        p=1;
        for q = 1:reps

            noise = squeeze(noise_matrix(q,:)).';

            [nclust_noise(q,p)] =...
                classified_zeros_detection(noise, 'estimate', J, {'kmeans','gap'} ,[2,2]);

            [signal_0, std_noise_0] = sigmerge(x,noise,0);
            [nclust_signal_0(q,p)] =...
                classified_zeros_detection(signal_0, 'estimate', J, {'kmans','gap'} ,[2,2]);

            [signal_5, std_noise_5] = sigmerge(x,noise,5);
            [nclust_signal_5(q,p)] =...
                classified_zeros_detection(signal_5, 'estimate', J, {'kmeans','gap'} ,[2,2]);

            [signal_10, std_noise_10] = sigmerge(x,noise,10);
            [nclust_signal_10(q,p)] =...
                classified_zeros_detection(signal_10, 'estimate', J, {'kmeans','gap'} ,[2,2]);

            %     disp(nclust_signal);
        end



        especificity_mean(k,j) = mean(sum(nclust_noise==1)/size(nclust_noise,1));
        especificity_std(k,j) = std(sum(nclust_noise==1)/size(nclust_noise,1));

        sensitivity_0_mean(k,j) = mean(sum(nclust_signal_0>1)/size(nclust_signal_0,1));
        sensitivity_0_std(k,j) = std(sum(nclust_signal_0>1)/size(nclust_signal_0,1));

        sensitivity_5_mean(k,j) = mean(sum(nclust_signal_5>1)/size(nclust_signal_5,1));
        sensitivity_5_std(k,j) = std(sum(nclust_signal_5>1)/size(nclust_signal_5,1));

        sensitivity_10_mean(k,j) = mean(sum(nclust_signal_10>1)/size(nclust_signal_10,1));
        sensitivity_10_std(k,j) = std(sum(nclust_signal_10>1)/size(nclust_signal_10,1));


    end
end


% Save results to finish the analysis in python.
detection_performance_results_matrix = [nclust_noise nclust_signal_0 nclust_signal_5 nclust_signal_10];
detection_performance_results_matrix(detection_performance_results_matrix==1) = 0;
detection_performance_results_matrix(detection_performance_results_matrix>0) = 1;


save new_results_det_perf_GMM.mat

%%
% 
load new_results_det_perf_GMM.mat
for i =1:length(Ns)
    plot(1-especificity_mean(i,:),sensitivity_0_mean(i,:)); hold on; %, 'DisplayName',['N=' string(N(i))]); hold on;
end