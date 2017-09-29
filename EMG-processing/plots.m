%% emg preprocessed
[emg,activation] = loadData([12:21]);

% for i=1:2
%     emg(i,:) = (emg(i,:) - mean(emg(i,:)))/std(emg(i,:));
% end;

emg = emg(2,:);
y = emg(1:50000);
x=[0:length(y)-1];
plot(y);

out = [x' y'];
save('../msc-thesis/graphics/prepr-emg-sample.txt', 'out',   '-ASCII' );

%% emg frequency domain
Fs = 600;            % Sampling frequency                    
clf
signal = y;

l = length(signal);
nfft2 = 2.^nextpow2(l);
fy = fft(signal,nfft2); % convert to frequency domain
fy = fy(1:nfft2/2);  % LHS of frequency signal 
xfft = Fs.*(0:nfft2/2 - 1)/nfft2;  % scale time to frequency domain

x = xfft;
y = (abs(fy)/max(fy));

out = [x' y'];
save('../msc-thesis/graphics/emg-frequency-domain.txt', 'out',   '-ASCII' );

% 
plot(xfft, abs(fy)/max(fy)); % normalized
% title(['Single Rectified Signal for ', str, ' in Frequency Domain']);
% ylabel('Normalized Magnitude');
% xlabel('Frequency (Hz)');

%% MVC
load('data/andrew8_calibration.mat')
emg = calib.calibration.biceps.EMG;
param.bandfilter = [10,500];
param.RMSwindow = 250;
Fs = 600;
param.sampleRate = Fs;
bfilt_emg =  bandfilter(emg',param.bandfilter(1),param.bandfilter(2),Fs);
% notch (50 Hz) 
emg = notch(bfilt_emg, param.sampleRate, 50);

% plot(emg)

x = 0:length(emg)-1;
y = emg;

RMS = nan(size(emg));
for j = param.RMSwindow:length(emg)-param.RMSwindow-1
    RMS(j,:) = rms(emg(j-param.RMSwindow+1:j+param.RMSwindow));
end

hold on;
% plot(RMS);

y1 = RMS;
out = [x' y y1];
save('../msc-thesis/graphics/mvc-rms.txt', 'out',   '-ASCII' );


%% DWT reconstruction
load('data/andrew9.mat');
emg = data.subject.biceps.EMG;
clf
sample = emg(102000:102800);

level = 4;
mother = 'db2';

[C , L] = wavedec(sample, level , mother);
A = appcoef(C , L , mother, level);
D = detcoef(C,L,level);

rec = idwt(A,D,mother);

subplot(211); plot(sample); 
title('Original signal'); 
subplot(212); plot(rec); 
title('Wavelet decomposition structure, level 4, db2 mother function') 
xlabel(['Coefs for approx. and det at level 4 '])
   
x = 1:length(sample);
y = sample;
x1 = 1:length(rec);
y1 = rec;
dim = length(sample);

x1(end+1:dim) = NaN;
y1(end+1:dim) = NaN;

out = [x' y x1' y1];
save('../msc-thesis/graphics/wavelet-rec-db2-level-4.txt', 'out',   '-ASCII' );


%% estimators' performance

load('lsq_dwt.mat')
load('nn_dwt.mat')
load('nn_rms.mat')

%% measured vs estimated activation

clf
dim = 20:320;
x = 0:length(dim)-1;
x=x*115/460;
y = nn_dwt.outputs(dim);
y1 = lsq_dwt.test_outputs(dim);
y2 = nn_dwt.test_outputs(dim);
y3 = nn_rms.test_outputs(dim);


% plot(nn_dwt.test_outputs);
% hold on;
% plot(nn_dwt.outputs);

out = [x' y y1 y2 y3];
save('../msc-thesis/graphics/meas-estim-act.txt', 'out',   '-ASCII' );

%% LSQ residual

y = lsq_dwt.residual;
x = 0:length(y)-1;
out = [x' y];
save('../msc-thesis/graphics/lsq-residual.txt', 'out',   '-ASCII' );

%% nn performance
p1 = nn_dwt.performance;
p2 = nn_rms.performance;

x1 = 0:p1.num_epochs;
x2 = 0:p2.num_epochs;

dim = length(x2);

y11 = p1.perf;
y12 = p1.vperf;
y13 = p1.tperf;

y21 = p2.perf;
y22 = p2.vperf;
y23 = p2.tperf;

x1(end+1:dim) = NaN;
y11(end+1:dim) = NaN;
y12(end+1:dim) = NaN;
y13(end+1:dim) = NaN;

plot(y11)

out = [x1' y11' y12' y13' x2' y21' y22' y23'];
save('../msc-thesis/graphics/nn-perf.txt', 'out',   '-ASCII' );

%% evaluation statistics

x = [1 2 3];
y1 = [lsq_dwt.rmse nn_dwt.rmse nn_rms.rmse];
y2 = [lsq_dwt.corrcoef(2) nn_dwt.corrcoef(2) nn_rms.corrcoef(2)];
y3 = [lsq_dwt.rsquared nn_dwt.rsquared nn_rms.rsquared];

out = [x' y3']';
save('../msc-thesis/graphics/eval-stats.txt', 'out',   '-ASCII' );

%%


