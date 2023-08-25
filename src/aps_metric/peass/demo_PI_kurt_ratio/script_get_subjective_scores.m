% Get Subjective Scores.
clear all;
load PEASS-subjdata.mat;

addpath('../');
addpath('../gammatone/');

scores_ArtificialNoiseAbsence = scores(:,IArtificialNoiseAbsenceScore);
scores_ArtificialNoiseAbsence = mean(scores_ArtificialNoiseAbsence);
file_names = string(soundNames);
file_names = file_names.';
file_names = file_names(IArtificialNoiseAbsenceScore);

output_names = [];
scores_AIA = [];
APS = [];
for i = [3,5,6,7,8,9]
    init = 5+8*(i-1);
    finit = init+3;
    archivos = [output_names; file_names(init:finit)];
    scores_AIA = [scores_AIA; scores_ArtificialNoiseAbsence(init:finit)];
    
    fstruct = dir(sprintf('exp0%d_InterfSrc*.wav',i));
    interf_files = struct2cell(fstruct);
    interf_files = interf_files(1,:);
    originalFiles = {sprintf('exp0%d_target.wav',i), interf_files{:}};
    
    [in_signal,fs] = generate_output(originalFiles);
    audiowrite(sprintf('exp0%d_input.wav',i),in_signal,fs)


    output_names = file_names(init:finit);
    aps = [];
    for j = 1:4
        estimateFile = [output_names{j} '.wav'];
        options.destDir = '../';
        options.segmentationFactor = 1; % increase this integer if you experienced "out of memory" problems
        res = PEASS_ObjectiveMeasure(originalFiles,estimateFile,options);
        aps = [aps res.APS];
    end
    APS = [APS; aps];

end

%%
figure()
for i = 1:4
    plot(scores_AIA(:,i),APS(:,i),'o'); hold on
end
xlim([0,100]);
ylim([20,90]);