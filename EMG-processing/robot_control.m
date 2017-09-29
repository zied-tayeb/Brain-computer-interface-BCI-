function acquisitionEMG()
    %% ros setup
    if not(robotics.ros.internal.Global.isNodeActive)
        rosinit('localhost');
    end

    %% load params 
    if not(exist('param', 'var'))
        setParam(loadParams());
        param = getParam();
    end

    %%
    move_motor = rossvcclient('/myo_blink/move');

    request = rosmessage(move_motor);

    request.Action = 'keep';
    min = 70;
    %max = 65;
    delta = randperm(param.trials*2);
    %forces = min:max;%unifrnd(min, max, 1, 2*param.trials);
    request.Setpoint = min;

    %% load params 
    if not(exist('param', 'var'))
        setParam(loadParams());
        param = getParam();
    end

    % input('Press ENTER to move the robot', 's');
    
    
    wrestle_topic = '/myo_blink/wrestle/start';
    wrestle_sub = rossubscriber(wrestle_topic);
    wrestle_pub = rospublisher(wrestle_topic,'std_msgs/Int32');
    receive(wrestle_sub);
    tic
    %% alternate force on the muscle
    i = 1;
    while (i <= param.trials*2)
        
        if (wrestle_sub.LatestMessage.Data == 0)
            break
        end
    %      try
    %         [emg_msg,~] = judp('RECEIVE',16571,400);
    %         
    %     catch
    %         warning('EMG data is not received. Robot actuation is on pause');
    % %         input('Press ENTER to continue: ', 's');
    %         continue;
    %      end

        if (i<=param.trials)
            muscle = 'biceps';
        else
            muscle = 'triceps';
        end
        request.Muscle = muscle;
        disp(min + delta(i));
        request.Setpoint = min + delta(i);%forces(i);
        disp('Wrestle');
        call(move_motor, request);
        pause(param.t_hold_force);    
        request.Setpoint = 38;
        call(move_motor, request);
        i = i + 1;
        disp('Relax');
        pause(param.t_relax);
    end

    toc
    
    msg = rosmessage(wrestle_pub);
    msg.Data = 0;
    send(wrestle_pub, msg);
    
    request.Setpoint = 38;
    request.Muscle = 'biceps';
    call(move_motor, request);
    request.Muscle = 'triceps';
    call(move_motor, request);
    
end



