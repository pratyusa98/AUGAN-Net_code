rng(42);

% 1. Define the parent folder path
% parentFolder = 'yaseen_khan/no_aug/denoise';
parentFolder = 'yaseen_khan/no_aug/Classification';

%parentFolder = 'physionet/train';
% parentFolder = 'physionet/val';

% 2. List subfolders inside the parent folder
subfolders = {'MR','AS', 'N','MS','MVP'};
% subfolders = {'abnormal','normal'};


% 3. Define STFT parameters
fftSize = 256;
hopLength = 32;
windowLength = 128;

% Specify the path to the 'han' folder
hanFolder = 'han_noisy';

% Get a list of noisy files from the 'han' folder
noisyFiles = dir(fullfile(hanFolder, '*.wav'));

% Loop through each subfolder
for subfolderIdx = 1:numel(subfolders)
    subfolderName = subfolders{subfolderIdx};
    
    % Create paths for clean_all and noisy_all data unseenfordenoise dataformodel
    % cleanAllFolder = fullfile('data/dataformodel_noaug/clean_stft_red-15db_Hristo');
    % noisyAllFolder = fullfile('data/dataformodel_noaug/noisy_stft_red-15db_Hristo');

    cleanAllFolder = fullfile('data/unseenfordenoise_noaug/test_clean_stft_red-15db_Hristo');
    noisyAllFolder = fullfile('data/unseenfordenoise_noaug/test_noisy_stft_red-15db_Hristo');

    
    % Create the subfolders if they don't exist
    if ~isfolder(cleanAllFolder)
        mkdir(cleanAllFolder);
    end
    if ~isfolder(noisyAllFolder)
        mkdir(noisyAllFolder);
    end
    
    % 4. Loop through audio files in the subfolder
    audioFiles = dir(fullfile(parentFolder, subfolderName, '*.wav'));
    for fileIdx = 1:numel(audioFiles)
        % Load the audio file
        audioFile = fullfile(audioFiles(fileIdx).folder, audioFiles(fileIdx).name);
        [y, fs] = audioread(audioFile);

        targetFs = 4000;
        y_resample = resample(y, targetFs, fs);
        y_resample_norm =  2 * (y_resample - min(y_resample)) / (max(y_resample) - min(y_resample)) - 1;

        % 5. Add noise to the audio
        desiredSNR = -15;  % Adjust the SNR if needed

        %wgn
        % noise = randn(size(y_resample_norm));
        % signal_power = rms(y_resample_norm)^2;
        % noise_power = signal_power / (10^(desiredSNR/10));
        % scaled_noise = sqrt(noise_power) * noise;
        % y_noisy = y_resample_norm + scaled_noise;

        % pinknoisy
        % noise = pinknoise(size(y_resample_norm),'like',y_resample_norm);
        % signalPower= sum(y_resample_norm.^2,1)/size(y_resample_norm,1);
        % noisePower = sum(noise.^2,1)/size(noise,1);
        % scaleFactor = sqrt(signalPower./(noisePower*(10^(desiredSNR/10))));
        % noise = noise.*scaleFactor;
        % y_noisy = y_resample_norm + noise;

        %rednoise
        noise = rednoise(length(y_resample_norm), 1);
        signal_power = rms(y_resample_norm)^2;
        noise_power = signal_power / (10^(desiredSNR/10));
        scaled_noise = sqrt(noise_power) * noise;
        y_noisy = y_resample_norm + scaled_noise;

    
        %hospital noisy
        % randNoisyIdx = randi(numel(noisyFiles));
        % noisyFile = fullfile(hanFolder, noisyFiles(randNoisyIdx).name);
        % [noisy, noisyFs] = audioread(noisyFile);
        % targetFs = 4000;
        % noisy_resampled = resample(noisy, targetFs, noisyFs);
        % heart_sound_length = length(y_resample_norm);
        % % noisy_signal_repeated = repmat(noisy, ceil(heart_sound_length / length(noisy)), 1);
        % noisy_signal = noisy_resampled(1:heart_sound_length);
        % y_noisy = y_resample_norm + noisy_signal;
        
        y_noisy_norm = 2 * (y_noisy - min(y_noisy)) / (max(y_noisy) - min(y_noisy)) - 1;
        
        % 6. Apply STFT to generate spectrogram for clean data
        anal_win = hann(windowLength, 'periodic');
        [STFT_clean, freq, t_stft_clean] = stft(y_resample_norm, anal_win, hopLength, fftSize, targetFs);
        
        % 7. Apply STFT to generate spectrogram for noisy data
        [STFT_noisy, ~, ~] = stft(y_noisy_norm, anal_win, hopLength, fftSize, targetFs);
        
        % 8. Divide the spectrogram into 128x64 segments with zero-padding
        segmentSize = [128, 64];
        numSegmentsTime = ceil(size(STFT_clean, 2) / segmentSize(2));
        numSegmentsFreq = floor(size(STFT_clean, 1) / segmentSize(1));
        for i = 1:numSegmentsFreq
            for j = 1:numSegmentsTime
                rowStart = (i - 1) * segmentSize(1) + 1;
                rowEnd = i * segmentSize(1);
                colStart = (j - 1) * segmentSize(2) + 1;
                colEndClean = min(j * segmentSize(2), size(STFT_clean, 2));
                colEndNoisy = min(j * segmentSize(2), size(STFT_noisy, 2));
                
                % Extract the segments
                segment_clean = STFT_clean(rowStart:rowEnd, colStart:colEndClean);
                segment_noisy = STFT_noisy(rowStart:rowEnd, colStart:colEndNoisy);
                
                % Zero-padding if needed
                if size(segment_clean, 2) < segmentSize(2)
                    segment_clean = [segment_clean, zeros(size(segment_clean, 1), segmentSize(2) - size(segment_clean, 2))];
                end
        
                if size(segment_noisy, 2) < segmentSize(2)
                    segment_noisy = [segment_noisy, zeros(size(segment_noisy, 1), segmentSize(2) - size(segment_noisy, 2))];
                end
        
                % Save the segments
                filename = sprintf('%s_%s_%d.mat', subfolderName,audioFiles(fileIdx).name(1:end-4), j);
                cleanFullFilePath = fullfile(cleanAllFolder, filename);
                noisyFullFilePath = fullfile(noisyAllFolder, filename);
        
                Segment_clean = segment_clean;
                Segment_noisy = segment_noisy;
        
                save(cleanFullFilePath, 'Segment_clean');
                save(noisyFullFilePath, 'Segment_noisy');
            end
        end
    end
end


%%

function [STFT, f, t] = stft(x, win, hop, nfft, fs)
% function: [STFT, f, t] = stft(x, win, hop, nfft, fs)
%
% Input:
% x - signal in the time domain
% win - analysis window function
% hop - hop size
% nfft - number of FFT points
% fs - sampling frequency, Hz
%
% Output:
% STFT - STFT-matrix (only unique points, time 
%        across columns, frequency across rows)
% f - frequency vector, Hz
% t - time vector, s
% representation of the signal as column-vector
x = x(:);
% determination of the signal length 
xlen = length(x);
% determination of the window length
wlen = length(win);
% stft matrix size estimation and preallocation
NUP = ceil((1+nfft)/2);     % calculate the number of unique fft points
L = 1+fix((xlen-wlen)/hop); % calculate the number of signal frames
STFT = zeros(NUP, L);       % preallocate the stft matrix
% STFT (via time-localized FFT)
for l = 0:L-1
    % windowing
    xw = x(1+l*hop : wlen+l*hop).*win;
    
    % FFT
    X = fft(xw, nfft);
    
    % update of the stft matrix
    STFT(:, 1+l) = X(1:NUP);
end
% calculation of the time and frequency vectors
t = (wlen/2:hop:wlen/2+(L-1)*hop)/fs;
f = (0:NUP-1)*fs/nfft;
end


function x = rednoise(m, n)
% input validation
validateattributes(m, {'double'}, ...
                      {'scalar', 'nonnan', 'nonempty', 'positive', 'finite', 'integer'}, ...
                      '', 'm', 1)
validateattributes(n, {'double'}, ...
                      {'scalar', 'nonnan', 'nonempty', 'positive', 'finite', 'integer'}, ...
                      '', 'n', 2) 
                  
% set the PSD slope
alpha = -2; 
% convert from PSD (power specral density) slope 
% to ASD (amplitude spectral density) slope
alpha = alpha/2;
% generate AWGN signal
x = randn(m, n);
% ensure that the processing is performed columnwise
% (this covers the case when one requests 1-by-n output matrix)
if m == 1, x = x(:); end
% calculate the number of unique fft points
NumUniquePts = ceil((size(x, 1)+1)/2);
% take fft of x
X = fft(x);
% fft is symmetric, throw away the second half
X = X(1:NumUniquePts, :);
% prepare a vector with frequency indexes 
k = 1:NumUniquePts; k = k(:);
% manipulate the first half of the spectrum so the spectral 
% amplitudes are proportional to the frequency by factor f^alpha
X = X.*(k.^alpha);
% perform ifft
if rem(size(x, 1), 2)	% odd length excludes Nyquist point 
    % reconstruct the whole spectrum
    X = [X; conj(X(end:-1:2, :))];
    
    % take ifft of X
    x = real(ifft(X));   
else                    % even length includes Nyquist point  
    % reconstruct the whole spectrum
    X = [X; conj(X(end-1:-1:2, :))];
    
    % take ifft of X
    x = real(ifft(X));  
end
% ensure zero mean value and unity standard deviation 
x = zscore(x);
% ensure the desired size of the output
% (this covers the case when one requests 1-by-n output matrix)
if m == 1, x = x'; end
end

