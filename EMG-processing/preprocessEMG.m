function RMS = preprocessEMG(mat)
    %% parameters
    param.bandfilter = [10,450]; % lower and upper freq
    param.lowfilter = 5;
    param.RMSwindow = 10;
    %param.nbframe = 4000; % number frame needed (interpolation)
    param.trials = 15;
    param.trialDuration = 2000;
    param.sampleRate = 500;
    param.freq = 500;
    % param.method = 'low' ; % RMS ou low
    
    %% load data
    tmp = load(mat);
    data = tmp.recording.X(:,1);
    
    %% treatment
    for i = 0:param.trials-1
        
        emg = data((i+1)*param.trialDuration - param.sampleRate + 1 : (i+1)*param.trialDuration);
        
        %emg = data(itrial).emg;

        % 1) Rebase
        norm_emg = emg - mean(emg);

        % 2) band-pass filter
        filt_emg =  bandfilter(norm_emg,param.bandfilter(1),param.bandfilter(2),param.freq);

        %) 3) signal rectification
        rect_emg = abs(filt_emg);

        % 4) low pass filter at 5Hz
        %filt_emg = lpfilter(rect_emg, param.lowfilter, param.freq);

        % 3) RMS
        RMS = nan(size(filt_emg));
        for j = param.RMSwindow:length(filt_emg)-param.RMSwindow-1
            RMS(j,:) = rms(filt_emg(j-param.RMSwindow+1:j+param.RMSwindow,:));
        end

        % 5) Normalization
        % emg = emg ./ (MVC/100);
    end
end
