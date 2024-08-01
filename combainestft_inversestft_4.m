
% Assign the loaded data to a variable (assuming the variable name in the .mat file is 'stft1')
% Load the .mat file
loadedData1 = load('test_noisy_data\augmented_New_AS_001_original_1.mat');

S1 = double(loadedData1.Segment_noisy);

% Load the second STFT .mat file
loadedData2 = load('test_noisy_data/augmented_New_AS_001_original_2.mat');

S2 = double(loadedData2.Segment_noisy);

% Load the second STFT .mat file
loadedData3 = load('test_noisy_data/augmented_New_AS_001_original_3.mat');

S3 = double(loadedData3.Segment_noisy);

% Load the second STFT .mat file
loadedData4 = load('test_noisy_data/augmented_New_AS_001_original_4.mat');

S4 = double(loadedData4.Segment_noisy);

% Load the second STFT .mat file
loadedData5 = load('test_noisy_data/augmented_New_AS_001_original_5.mat');

S5 = double(loadedData5.Segment_noisy);

% 
% % Load the second STFT .mat file
% loadedData5 = load('denoised_pcg_sound_test5.mat');
% 
% S5 = double(loadedData4.S_real);



%%


% Merge the two STFT matrices to create a (128, 128) matrix
merged_stft = [S1,S2,S3,S4,S5];

%save('spectrogram_hnoise_unet_fullNew_MR.mat', 'merged_stft');

%%

% Define the folder containing the .mat files
mat_folder = 'test_noisy_data';

% Get a list of all .mat files in the folder
mat_files = dir(fullfile(mat_folder, '*.mat'));

% Initialize an empty cell array to store loaded data
loaded_data = cell(1, numel(mat_files));

% Loop through all .mat files in the folder
for i = 1:numel(mat_files)
    % Construct the full path to the .mat file
    full_path = fullfile(mat_folder, mat_files(i).name);

    % Load the .mat file
    loaded_data{i} = load(full_path);
end

% Extract the Segment_clean data from each loaded file
segments_clean = cellfun(@(x) double(x.Segment_clean), loaded_data, 'UniformOutput', false);

% Concatenate all loaded data along the second dimension
merged_stft = cat(2, segments_clean{:});

%%

% loadedData = load('data_classi\data_noaug\han\clean\New_MS_185.mat');

% S = double(loadedData.denoise_spe);


% % Define STFT parameters
fftSize = 256;
hopLength = 32;
windowLength = 128;

% generate analysis and synthesis windows
anal_win = hann(windowLength, 'periodic');
synth_win = hann(windowLength, 'periodic');

% inverse STFT

[y_reconstruct, t_reconstruct] = istft(merged_stft, anal_win, synth_win, hopLength, fftSize, 1000);
y_reconstruct_normalized = 2 * (y_reconstruct - min(y_reconstruct)) / (max(y_reconstruct) - min(y_reconstruct)) - 1;

figure;
plot(t_reconstruct(1:4995), y_reconstruct_normalized(1:4995)');
xlabel('Time (s)');
ylabel('Amplitude');
title('Reconstuct Sound Data');


%%


% Read the audio data and sample rate
[audioData, Fs] = audioread('Yaseen_Khan\MR\New_MR_195.wav');

% Calculate the time vector based on the sample rate
time = (0:length(audioData)-1) / Fs;

% Resample to 4000 Hz
targetFs = 4000;
y_resampled = resample(audioData, targetFs, Fs);

% Create a time vector for plotting
t_resmaple = (0:length(y_resampled)-1) / targetFs;

figure;
plot(t_resmaple,y_resampled);
xlabel('Time (s)');
ylabel('Amplitude');

title('Original Heart Sound');


%%

% Load the denoise .mat file
loadedData = load('New_AS_002_denoise_model0001_aug_cv250.mat');

denoise_unett = double(loadedData.denoise_spe);

% realPart = imag(loadedData.absSegment_noisy);

% doing inverse stft obtain clean signal

% % Define STFT parameters
fftSize = 256;
hopLength = 32;
windowLength = 128;

% generate analysis and synthesis windows
anal_win = hann(windowLength, 'periodic');
synth_win = hann(windowLength, 'periodic');

% inverse STFT

[y_reconstruct, t_reconstruct] = istft(denoise_unett, anal_win, synth_win, hopLength, fftSize, 4000);


% Plot the lung sound data
figure;
plot(t_reconstruct, y_reconstruct');
xlabel('Time (s)');
ylabel('Amplitude');
title('Reconstuct Heart Sound Aug');


%%

% https://in.mathworks.com/matlabcentral/fileexchange/-
% 45577-inverse-short-time-fourier-transform-istft-with-matlab

function [x, t] = istft(STFT, awin, swin, hop, nfft, fs)
% function: [x, t] = istft(STFT, awin, swin, hop, nfft, fs)
%
% Input:
% stft - STFT-matrix (only unique points, time
%        across columns, frequency across rows)
% awin - analysis window function
% swin - synthesis window function
% hop - hop size
% nfft - number of FFT points
% fs - sampling frequency, Hz
%
% Output:
% x - signal in the time domain
% t - time vector, s
% signal length estimation and preallocation
L = size(STFT, 2);          % determine the number of signal frames
wlen = length(swin);        % determine the length of the synthesis window
xlen = wlen + (L-1)*hop;    % estimate the length of the signal vector
x = zeros(1, xlen);         % preallocate the signal vector
% reconstruction of the whole spectrum
if rem(nfft, 2)             
    % odd nfft excludes Nyquist point
    X = [STFT; conj(flipud(STFT(2:end, :)))];
else                        
    % even nfft includes Nyquist point
    X = [STFT; conj(flipud(STFT(2:end-1, :)))];
end
% columnwise IFFT on the STFT-matrix
xw = real(ifft(X));
xw = xw(1:wlen, :);
% Weighted-OLA
for l = 1:L
    x(1+(l-1)*hop : wlen+(l-1)*hop) = x(1+(l-1)*hop : wlen+(l-1)*hop) + ...
                                      (xw(:, l).*swin)';
end
% scaling of the signal
W0 = sum(awin.*swin);                  
x = x.*hop/W0;                      
% generation of the time vector
t = (0:xlen-1)/fs;                 
end