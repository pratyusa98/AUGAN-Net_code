
%here use LMS, Threshold and Unet

%%
% Read the audio data and sample rate
[audioData, Fs] = audioread("Yaseen_Khan\MS\New_MS_185.wav");

% Calculate the time vector based on the sample rate
time = (0:length(audioData)-1) / Fs;

% Resample to 2000 Hz
targetFs = 4000;
y_resampled = resample(audioData, targetFs, Fs);

% Create a time vector for plotting
t = (0:length(y_resampled)-1) / targetFs;

% Plot the lung sound data
figure;
plot(t, y_resampled);
xlabel('Time (s)');
ylabel('Amplitude');
title('Heart Murmur Sound Data');

%%
rng(42);
desiredSNR = -5;  % Adjust the SNR if needed

%wgn
noise = wgn(length(y_resampled), 1, desiredSNR);
y_noisy = y_resampled + noise;

y_noisy_normalized = y_noisy / max(abs(y_noisy));

%filename = 'augmented_New_AS_162_original.wav';
%audiowrite(filename, y_noisy_normalized, targetFs); 

figure;
plot(y_noisy);
xlabel('Time (s)');
ylabel('Amplitude');
title('Original Sound Data');

%%

noise = randn(size(y_resampled));
signal_power = rms(y_resampled)^2;
noise_power = signal_power / (10^(desiredSNR/10));
scaled_noise = sqrt(noise_power) * noise;
x_with_noise = y_resampled + scaled_noise;
figure;
plot(x_with_noise);
xlabel('Time (s)');
ylabel('Amplitude');
title('Original Sound Data');

%%

noised = pinknoise(size(y_resampled),'like',y_resampled);
signalPower= sum(y_resampled.^2,1)/size(y_resampled,1);
noisePower = sum(noised.^2,1)/size(noised,1);
scaleFactor = sqrt(signalPower./(noisePower*(10^(desiredSNR/10))));
noised = noised.*scaleFactor;
y_noisyi = y_resampled + noised;

%%

%other noise

[noisy, noisyFs] = audioread('han/seg_3.wav');

t_noisy = (0:length(noisy)-1) / noisyFs;

% Determine the length of the resampled heart sound signal
heart_sound_length = length(y_resampled);

% Repeat the noisy signal to match the length of the heart sound signal
noisy_signal_repeated = repmat(noisy, ceil(heart_sound_length / length(noisy)), 1);

% Trim the repeated noisy signal to match the length of the heart sound signal
noisy_signal_repeated = noisy_signal_repeated(1:heart_sound_length);

% Add the repeated noisy signal to the resampled heart sound signal
y_noisy = y_resampled + noisy_signal_repeated;

% 
figure;
subplot(3,1,1)
plot(t, y_resampled);
xlabel('Time (s)');
ylabel('Amplitude');
title('Original Sound Data');

subplot(3,1,2)
plot(t_noisy, noisy);
xlabel('Time (s)');
ylabel('Amplitude');
title('External Noise');


subplot(3,1,3)
plot(t, y_noisy);
title('Combined Heart Sound and Noisy Signal');
xlabel('Time (s)');
ylabel('Amplitude');


%%


% Define the LMS filter parameters

filterLength = 5; % Filter length
mu = 0.0005; % LMS step size

% Initialize the LMS filter
lmsFilter = dsp.LMSFilter('Length', filterLength, 'StepSize', mu);

% Apply the LMS filter to denoise the signal
[denoisedSignal, errorSignal] = step(lmsFilter, y_resampled, y_noisy);

plot(t, denoisedSignal);
title('Denoised Signal');
xlabel('Time (s)');



%%


% 'SURE', 'Bayes', 'UniversalThreshold', 'FDR', 'Minimax', 'BlockJS'

% Perform wavelet denoising using RigSure thresholding
y_denoised_sure = wdenoise(y_noisy,5, ...
    Wavelet='db4', ...
    DenoisingMethod='SURE', ...
    ThresholdRule='Soft');


plot(t, y_denoised_sure);
xlabel('Time (s)');
ylabel('Amplitude');
title('Denoise Signal Using SURE Threshold Method');



%%


% % Define the wavelet and decomposition level
% wavelet = 'db12';  % You can choose a different wavelet
% level = 10;         % Adjust the level as needed
% 
% % Perform DWT decomposition
% [c, l] = wavedec(y_noisy, level, wavelet);
% 
% % Estimate the noise standard deviation using the universal threshold rule
% sigma = mad(c) / 0.6745;
% 
% % Define the threshold for denoising (you can adjust this threshold)
% threshold = sigma * sqrt(2 * log(length(y_noisy)));
% 
% % Apply soft thresholding to the DWT coefficients
% c_denoised = wthresh(c, 's', threshold);
% 
% % Reconstruct the denoised signal
% y_clean = waverec(c_denoised, l, wavelet);
% 
% 
% % Plot the original noisy signal and the cleaned signal
% figure;
% subplot(3,1,1);
% plot(t, y_resampled);
% title('Original Signal');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(3,1,2);
% plot(t, y_noisy);
% title('Noisy Signal');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(3,1,3);
% plot(t, y_clean);
% title('Cleaned Signal');
% xlabel('Time (s)');
% ylabel('Amplitude');

%%


% % Define STFT parameters
fftSize = 256;
hopLength = 32;
windowLength = 128;

% generate analysis and synthesis windows
anal_win = hann(windowLength, 'periodic');
synth_win = hann(windowLength, 'periodic');


% Load the .mat file
loadedData = load('spectrogram_hnoise_unet_fullNew_MR.mat');

S2 = double(loadedData.merged_stft);

% inverse STFT

[y_reconstruct, t_reconstruct] = istft(S2, anal_win, synth_win, hopLength, fftSize, 4000);

%%


% Plot the original noisy signal and the cleaned signal
figure();
subplot(5,1,1);
plot(t(1:7200), y_resampled(1:7200));
title('Original Abnormal Signal');

subplot(5,1,2);
plot(t(1:7200), y_noisy(1:7200));
title('Noise Added Signal (Hospital Ambient Noise)');

subplot(5,1,3)
plot(t(1:7200), y_denoised_sure(1:7200));
ylabel('Amplitude');
title('Denoise Signal Using SURE Threshold Method');

subplot(5,1,4)
plot(t(1:7200), denoisedSignal(1:7200));
title('Denoised Signal Using LMS Filter');

subplot(5,1,5)
plot(t_reconstruct(1:7200), y_reconstruct(1:7200));
xlabel('Time (s)');
title('Denoise Signal using Unet Method after 100 Epoch');



%%


% Calculate the power of the original signal
signal_power = sum(audioData.^2);

% Calculate the power of the noise
noise_power = sum((y_resampled - denoisedSignal).^2);

% Calculate the SNR in dB
snr_db = 10 * log10(signal_power / noise_power);

fprintf('SNR (dB): %.2f\n', snr_db);

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

