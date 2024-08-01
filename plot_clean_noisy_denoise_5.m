%for Lunet
% Read the WAV file
[y, Fs] = audioread('Yaseen_Khan\no_aug\classification\MR\New_MR_188.wav');

% Resample to 1 kHz
Fs_new = 1000;
y_resampled = resample(y, Fs_new, Fs);

y_resample_norm =  2 * (y_resampled - min(y_resampled)) / (max(y_resampled) - min(y_resampled)) - 1;

desiredSNR = -5;

% noise = randn(size(y_resample_norm));
% signal_power = rms(y_resample_norm)^2;
% noise_power = signal_power / (10^(desiredSNR/10));
% scaled_noise = sqrt(noise_power) * noise;
% y_with_noise = y_resample_norm + scaled_noise;

% noise = pinknoise(size(y_resample_norm),'like',y_resample_norm);
% signalPower= sum(y_resample_norm.^2,1)/size(y_resample_norm,1);
% noisePower = sum(noise.^2,1)/size(noise,1);
% scaleFactor = sqrt(signalPower./(noisePower*(10^(desiredSNR/10))));
% noise = noise.*scaleFactor;
% y_with_noise = y_resample_norm + noise;

% 
noise = rednoise(length(y_resample_norm), 1);
signal_power = rms(y_resample_norm)^2;
noise_power = signal_power / (10^(desiredSNR/10));
scaled_noise = sqrt(noise_power) * noise;
y_with_noise = y_resample_norm + scaled_noise;


segment_duration = 0.8; % in seconds
segment_samples = segment_duration * Fs_new;
num_segments = floor(length(y_with_noise) / segment_samples);

% Determine the remaining samples after segmenting
remaining_samples = mod(length(y_with_noise), segment_samples);
if remaining_samples > 0
    % Calculate the number of zero samples to pad
    zero_padding = segment_samples - remaining_samples;
    % Add zero padding to the last segment
    y_with_noise = [y_with_noise; zeros(zero_padding, 1)];
    % Update the number of segments
    num_segments = num_segments + 1;
end

% Define the folder path where segmented files will be stored
folder_path = 'Lu_net_denoise/segmented_files';

% Check if the folder exists
if exist(folder_path, 'dir')
    % If the folder exists, delete it and its contents
    rmdir(folder_path, 's');
end

% Create the folder again
mkdir(folder_path);

for i = 1:num_segments
    segment = y_with_noise((i-1)*segment_samples + 1 : i*segment_samples);
    segment_filename = sprintf('%s/segment_%d.mat', folder_path, i);
    save(segment_filename, 'segment');
end

disp('Segments saved successfully.');


%%
clear all;
clc;

% % Define STFT parameters
fftSize = 256;
hopLength = 32;
windowLength = 128;

% generate analysis and synthesis windows
anal_win = hann(windowLength, 'periodic');
synth_win = hann(windowLength, 'periodic');


name = 'AS';  % Replace 'YourName' with the actual value
sampleno = 198;     % Replace 123 with the actual value augmented_New_AS_161_shift_1

loadedData_clean =  load(sprintf('data_classi\\data_noaug\\han\\clean\\New_%s_%d.mat', name, sampleno));
loadedData_clean = double(loadedData_clean.denoise_spe);
[y_clean, t_clean] = istft(loadedData_clean, anal_win, synth_win, hopLength, fftSize, 4000);

 
loadedData_noisy = load(sprintf('data_classi\\data_noaug\\han\\noisy\\New_%s_%d.mat', name, sampleno));
loadedData_noisy = double(loadedData_noisy.denoise_spe);
[y_noisy, t_noisy] = istft(loadedData_noisy, anal_win, synth_win, hopLength, fftSize, 4000);

% loadedData_denoisy = load('New_AS_002_denoise_model0001_aug_cv250.mat');
loadedData_denoisy_dwt = load(sprintf('data_classi\\data_noaug\\han\\denoise_dwt\\New_%s_%d.mat', name, sampleno));
loadedData_denoisy_dwt = double(loadedData_denoisy_dwt.denoise_spe);
[y_denoisy_dwt, t_denoisy_dwt] = istft(loadedData_denoisy_dwt, anal_win, synth_win, hopLength, fftSize, 4000);

% loadedData_denoisy = load('New_AS_002_denoise_model0001_aug_cv250.mat');
% loadedData_denoisy_unet = load(sprintf('data_classi\\data_noaug\\red-5\\denoise_unet_mix_avg\\New_%s_%d.mat', name, sampleno));
% loadedData_denoisy_unet = double(loadedData_denoisy_unet.denoise_spe);
% [y_denoisy_unet, t_denoisy_unet] = istft(loadedData_denoisy_unet, anal_win, synth_win, hopLength, fftSize, 4000);


% loadedData_denoisy = load('New_AS_002_denoise_model0001_aug_cv250.mat');
loadedData_denoisy_unet = load(sprintf('data_classi\\data_noaug\\han\\denoise_unet\\New_%s_%d.mat', name, sampleno));
loadedData_denoisy_unet  = double(loadedData_denoisy_unet.denoise_spe);
[y_denoisy_unet , t_denoisy_unet ] = istft(loadedData_denoisy_unet, anal_win, synth_win, hopLength, fftSize, 4000);

loadedData_denoisy_gan = load(sprintf('data_classi\\data_noaug\\han\\denoise_gan\\New_%s_%d.mat', name, sampleno));
loadedData_denoisy_gan = double(loadedData_denoisy_gan.denoise_spe);
[y_denoisy_gan, t_denoisy_gan] = istft(loadedData_denoisy_gan, anal_win, synth_win, hopLength, fftSize, 4000);

loadedData_denoisy_attnunet = load(sprintf('data_classi\\data_noaug\\han\\denoise_unet_mix\\New_%s_%d.mat', name, sampleno));
loadedData_denoisy_attnunet = double(loadedData_denoisy_attnunet.denoise_spe);
[y_denoisy_attnunet, t_denoisy_attnunet] = istft(loadedData_denoisy_attnunet, anal_win, synth_win, hopLength, fftSize, 4000);

loadedData_denoisy_ganattunet = load(sprintf('data_classi\\data_noaug\\han\\denoise_unetattn_p\\New_%s_%d.mat', name, sampleno));
loadedData_denoisy_ganattunet = double(loadedData_denoisy_ganattunet.denoise_spe);
[y_denoisy_ganattunet, t_denoisy_ganattunet] = istft(loadedData_denoisy_ganattunet, anal_win, synth_win, hopLength, fftSize, 4000);


% Load the full signal from the .mat file Lunet
data = load('Lu_net_denoise/full_signal.mat');
full_signal = data.full_signal;
% Generate time vector
Fs = 1000; % Sampling frequency in Hz
t = (0:length(full_signal)-1) / Fs; % Time vector



y_clean_normalized = 2 * (y_clean - min(y_clean)) / (max(y_clean) - min(y_clean)) - 1;
y_noisy_normalized = 2 * (y_noisy - min(y_noisy)) / (max(y_noisy) - min(y_noisy)) - 1;
y_denoisy_dwt_normalized = 2 * (y_denoisy_dwt - min(y_denoisy_dwt)) / (max(y_denoisy_dwt) - min(y_denoisy_dwt)) - 1;
y_denoisy_unet_normalized = 2 * (y_denoisy_unet - min(y_denoisy_unet)) / (max(y_denoisy_unet) - min(y_denoisy_unet)) - 1;
y_denoisy_gan_normalized = 2 * (y_denoisy_gan - min(y_denoisy_gan)) / (max(y_denoisy_gan) - min(y_denoisy_gan)) - 1;
data_norm = 2 * (full_signal - min(full_signal)) / (max(full_signal) - min(full_signal)) - 1;

y_denoisy_attnunet_normalized = 2 * (y_denoisy_attnunet - min(y_denoisy_attnunet)) / (max(y_denoisy_attnunet) - min(y_denoisy_attnunet)) - 1;
y_denoisy_ganattunet_normalized = 2 * (y_denoisy_ganattunet - min(y_denoisy_ganattunet)) / (max(y_denoisy_ganattunet) - min(y_denoisy_ganattunet)) - 1;

% Assuming y_denoisy_normalized and y_noisy_normalized are your sequences


% Define the lengths of the two arrays
% len_y_noisy = length(y_noisy_normalized);
% len_y_denoisy = length(y_denoisy_normalized);
% 
% % Calculate the amount of padding needed
% padding_needed = len_y_noisy - len_y_denoisy;
% 
% % Pad y_denoisy_normalized with zeros
% if padding_needed > 0
%     y_denoisy_normalize = padarray(y_denoisy_normalized, [0, padding_needed], 0, 'post');
% end


% 
figure;
subplot(8,1,1)
plot(t_clean, y_clean_normalized);
title('Clean Signal');
xlim([0,2]);
set(gca,'XTickLabel',[]);

subplot(8,1,2)
plot(t_noisy, y_noisy_normalized);
title('Noise Signal');
xlim([0,2]);
set(gca,'XTickLabel',[]);

subplot(8,1,3)
plot(t_denoisy_dwt, y_denoisy_dwt_normalized);
title('Denoised Signal DWT');
xlim([0,2]);
set(gca,'XTickLabel',[]);

subplot(8,1,4)
plot(t, data_norm);
title('Denoised Signal LUnet');
xlim([0,2]);
set(gca,'XTickLabel',[]);

subplot(8,1,5)
plot(t_denoisy_unet, y_denoisy_unet_normalized);
title('Denoised Signal Unet');
xlim([0,2]);
set(gca,'XTickLabel',[]);

subplot(8,1,6)
plot(t_denoisy_gan, y_denoisy_gan_normalized);
title('Denoised Signal GAN');
xlim([0,2]);
set(gca,'XTickLabel',[]);

subplot(8,1,7)
plot(t_denoisy_attnunet, y_denoisy_attnunet_normalized);
title('Denoised Signal Attnunet');
xlim([0,2]);
set(gca,'XTickLabel',[]);

subplot(8,1,8)
plot(t_denoisy_ganattunet, y_denoisy_ganattunet_normalized);
title('Denoised Signal GanAttnunet');
% xlabel('Time (s)');
% ylabel('Amplitude');
xlim([0,2]);

% sgtitle('Denoised Signal for Red -5 dbnoise');
%saveas(gcf, 'C:\Users\Admin\OneDrive - National Institute of Technology, Rourkela\Desktop\ima\han\MVP_178.png');

% disp(10 * log10(sum(y_clean_normalized.^2) / sum((y_denoisy_normalized - y_clean_normalized).^2)));

%disp(calculateSISDR(y_clean,y_denoisy));

%%

% Define parameters
name = {'AS', 'MS', 'MVP', 'N','MR'};
sampleno_range = 161:200;
clean_folder_path = 'data_classi\data_noaug\red-5\clean\';
denoise_folder_path = 'data_classi\data_noaug\red-5\denoise_attnresi_Hristo\';

fftSize = 256; % Specify your desired FFT size
hopLength = 32; % Adjust according to your analysis needs
windowLength = 128;
% generate analysis and synthesis windows
anal_win = hann(windowLength, 'periodic');
synth_win = hann(windowLength, 'periodic');

% Initialize arrays to store SNR values
snr_values = zeros(length(name), length(sampleno_range));
mean_snr_values = zeros(length(name), 1);

% Loop through each folder
for i = 1:length(name)
    fprintf('\nName: %s\n', name{i});
    for j = 1:length(sampleno_range)
        % Load clean and denoised signals
        clean_file = fullfile(clean_folder_path, sprintf('New_%s_%d.mat', name{i}, sampleno_range(j)));
        denoise_file = fullfile(denoise_folder_path, sprintf('New_%s_%d.mat', name{i}, sampleno_range(j)));

        loadedData_clean = load(clean_file);
        loadedData_denoisy = load(denoise_file);


        % Perform inverse STFT
        [y_clean, t_clean] = istft(double(loadedData_clean.denoise_spe), anal_win, synth_win, hopLength, fftSize, 4000);
        [y_denoisy, t_denoisy] = istft(double(loadedData_denoisy.denoise_spe), anal_win, synth_win, hopLength, fftSize, 4000);

        % Normalize signals
        y_clean_normalized = 2 * (y_clean - min(y_clean)) / (max(y_clean) - min(y_clean)) - 1;
        y_denoisy_normalized = 2 * (y_denoisy - min(y_denoisy)) / (max(y_denoisy) - min(y_denoisy)) - 1;

        % Calculate SNR 
        snr_value = compute_si_sdr(y_clean,y_denoisy);
        snr_values(i, j) = snr_value;

        % Display SNR for each sample
        fprintf("{'Name': '%s', 'SampleNo': %d, 'SNR': %.2f}\n", name{i}, sampleno_range(j), snr_value);
    end
    % Calculate mean SNR for each folder
    mean_snr_values(i) = mean(snr_values(i, :));
end

% % Display mean SNR values for each class
% fprintf('\nMean SNR Values for Each Class:\n');
% for i = 1:length(name)
%     fprintf("{'Name': '%s', 'MeanSNR': %.2f}\n", name{i}, mean_snr_values(i));
% end

% Calculate overall mean SNR
overall_mean_snr = mean(mean_snr_values);
fprintf('\nOverall Mean SNR: %.2f\n', overall_mean_snr);



%%

function si_sdr = compute_si_sdr(reference, estimate)
    % Check if the two signals have the same length
    % if length(reference) ~= length(estimate)
    %     error('The reference and estimate signals must have the same length.');
    % end

    % Convert the signals to column vectors
    reference = reference(:);
    estimate = estimate(:);

    % Compute the optimal scale for the estimate to minimize distortion
    alpha = (reference' * estimate) / (reference' * reference);

    % Compute the projection of the estimate onto the reference
    projection = alpha * reference;

    % Compute the distortion (error)
    error = estimate - projection;

    % Compute signal power and distortion power
    signal_power = sum(projection .^ 2);
    distortion_power = sum(error .^ 2);

    % Compute SI-SDR in dB
    si_sdr = 10 * log10(signal_power / distortion_power);
end



% function sisdr_value = calculateSISDR(target_signal, estimated_signal)
%     % Ensure that the input signals are column vectors
%     target_signal = target_signal(:);
%     estimated_signal = estimated_signal(:);
% 
%     % Calculate SISDR
%     sisdr_value = 10 * log10(sum(target_signal.^2) / sum((target_signal - estimated_signal).^2));
% end

%%

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