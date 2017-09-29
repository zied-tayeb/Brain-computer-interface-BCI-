function [filt_emg] = filterEMG(emg,MVC)
param = loadParams();

% band-pass filter
bfilt_emg =  bandfilter(emg,param.bandfilter(1),param.bandfilter(2),param.freq);


% notch (50 Hz) 
nfilt_emg = notch(bfilt_emg, param.sampleRate, 50);

% high-pass
hfilt_emg = hpfilter(nfilt_emg, param.highfilter, param.sampleRate);

% signal rectification
rect_emg = abs(hfilt_emg);

% low pass filter at 5Hz
lfilt_emg = lpfilter(rect_emg, param.lowfilter, param.freq);

if isnan(MVC)
    % 5) RMS
    RMS = nan(size(lfilt_emg));
    for j = param.RMSwindow:length(lfilt_emg)-param.RMSwindow-1
        RMS(j,:) = rms(lfilt_emg(j-param.RMSwindow+1:j+param.RMSwindow,:));
    end
    filt_emg = RMS;
else
    % 5) Normalization
    filt_emg = lfilt_emg ./ (MVC*0.8/100);
    
end

t = 1:length(emg);
%plot(t, bfilt_emg, t, nfilt_emg, t, hfilt_emg, t, rect_emg, t, lfilt_emg, t, filt_emg)
%legend('bandpass', 'notch', 'highpass', 'rectified', 'lowpass', 'normalized');

plot( t, filt_emg)
legend('normalized');

end