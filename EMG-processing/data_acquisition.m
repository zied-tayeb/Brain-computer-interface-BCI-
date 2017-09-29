function [emg_array] = saveSubjectData()
    session = input('session name: ', 's');
    data.subject.name = input('name: ', 's');
    data.subject.age = input('age: ', 's');
    data.subject.height = input('height: ', 's');
    data.subject.weight = input('weight: ', 's');

    %% ros setup
    if not(robotics.ros.internal.Global.isNodeActive)
        rosinit('localhost');
    end

    %% load params 
    if not(exist('param', 'var'))
        setParam(loadParams());
        param = getParam();
    end
    
    disp('Loading the library...');
    lib = lsl_loadlib();

    % resolve a stream...
    disp('Resolving an EEG stream...');
    result = {};
    while isempty(result)
        result = lsl_resolve_byprop(lib,'type','EEG'); end

    % create a new inlet
    disp('Opening an inlet...');
    inlet = lsl_inlet(result{1});

    %% allocate space 
    muscles = [string('biceps'), string('triceps')];
    dim = param.sampleRate * (2 * param.trials * (param.t_hold_force + param.t_relax));

    for i=1:length(muscles)
        emg_array.(muscles{i}) = nan(1,dim);
        muscle_msgs.(muscles{i}) = robotics.ros.custom.msggen.myo_blink.muscleState.empty(dim,0);
        data.robot.(muscles{i}).force = nan(1,dim);
        data.robot.(muscles{i}).l_CE = nan(1,dim);
        data.robot.(muscles{i}).delta_l_SEE = nan(1,dim);
        data.robot.(muscles{i}).dot_l_CE = nan(1,dim);
        data.robot.(muscles{i}).dot_l_SEE = nan(1,dim);
        subscriber.(muscles{i}) = rossubscriber( strcat('/myo_blink/muscles/', muscles{i}, '/sensors'));
    end
    data.robot.joint_angles = nan(1,dim);
%     data.subject.timestamp = nan(1,dim);

    %%
    wrestle_topic = '/myo_blink/wrestle/start';
    wrestle_sub = rossubscriber(wrestle_topic);
    %% obtain MVC
%     [calibration.biceps.MVC, calibration.biceps.EMG] = calculateMVC('biceps');
%     [calibration.triceps.MVC, calibration.triceps.EMG] = calculateMVC('triceps');
% 
%     calib = struct('calibration', calibration);
%     save(strcat(session, '_calibration', '.mat'), 'calib');
    % input('Calibration completed. Pres ENTER to continue the experiment', 's');

    receive(wrestle_sub);
    disp('saving data');
    tic
    %%
    j=1;
    while (wrestle_sub.LatestMessage.Data == 1)
        try
%             [emg_msg,~] = judp('RECEIVE',16571,400);
%             emg = jsondecode(char(emg_msg));
            [emg,~] = inlet.pull_sample();
        catch
            warning('Corrupted EMG data, skipping the message');
            continue;
        end

        for i=1:length(muscles)
            ch = param.channels(char(muscles(i)));
            emg_array.(muscles{i})(j) = emg(ch(2)) -  emg(ch(1));
            try
                msg = subscriber.(muscles{i}).LatestMessage;
                muscle_msgs.(muscles{i})(j) = msg;
            catch
                warning('incomplete data from ROS');
                continue;
            end
        end
        
        
        j=j+1;    
    end
    data.time=toc;
    
    %% extract robot state data
    for k=1:j-1
        for i=1:length(muscles)
            msg = muscle_msgs.(muscles{i})(k);
            try
                data.robot.(muscles{i}).force(k) = msg.ElasticDisplacement * 0.2 + 38;
        %         raw values of the robot state, no conversion
                data.robot.(muscles{i}).l_CE(k) = msg.ContractileDisplacement;% * 0.006 * pi / 6.28319;
                data.robot.(muscles{i}).delta_l_SEE(k) = msg.ElasticDisplacement;
                data.robot.(muscles{i}).dot_l_CE(k) = msg.ActuatorVel;
                data.robot.(muscles{i}).dot_l_SEE(k) = msg.ElasticVel;
            catch
                warning('incomplete data from ROS');
                continue;
            end
    
        end
    end
    
    %% clean data from NaN   
    for i=1:length(muscles)
        del = isnan(emg_array.(muscles{i}));
        emg_array.(muscles{i})(del) =[];
        data.robot.(muscles{i}).force(del) = [];
        data.robot.(muscles{i}).l_CE(del) = [];
        data.robot.(muscles{i}).delta_l_SEE(del) = [];
        data.robot.(muscles{i}).dot_l_CE(del) = [];
        data.robot.(muscles{i}).dot_l_SEE(del) = [];
        
    end
    
   
    data.robot.joint_angles(del) = [];
    
    emg_array.biceps(isnan(emg_array.biceps)) = [];
    fr = length(emg_array.biceps)/data.time
    
    
    %% apply basic filters 
    disp('preprocessing data..');
    clear emg
    fields = fieldnames(emg_array);
    for i = 1:numel(fields)
        emg = emg_array.(fields{i});
        % band-pass
        bfilt_emg =  bandfilter(emg',param.bandfilter(1),param.bandfilter(2),param.freq);
        % notch (50 Hz) 
        data.subject.(fields{i}).EMG = notch(bfilt_emg, param.sampleRate, 50);
    end
    %%
    dim = 1:length(data.subject.biceps.EMG);
    plot(data.subject.biceps.EMG);
%     plot(dim, data.subject.triceps.EMG, dim, data.robot.biceps.force)

    save(strcat(session, '.mat'), 'data');
end