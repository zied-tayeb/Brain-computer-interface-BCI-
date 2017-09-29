if not(robotics.ros.internal.Global.isNodeActive)
    rosinit('localhost');
end

[emg,activation] = loadData([17:21]);
for i=1:2
    emg(i,:) = (emg(i,:) - mean(emg(i,:)))/std(emg(i,:));
end;

% [trainInd,valInd,testInd] = divideblock(length(emg),0.1,0.9,0);

testemg = emg(2,valInd);

testa = activation(valInd);

dim=length(testemg);
pub = rospublisher('/emg','std_msgs/Float32');
emg_msg = rosmessage(pub);

real_act_pub = rospublisher('/activation/real','std_msgs/Float32');
a_msg = rosmessage(real_act_pub);

r = robotics.Rate(600);
reset(r);
for i=1:dim
    emg_msg.Data = testemg(i);
    a_msg.Data = testa(i);
    send(pub,emg_msg);
    send(real_act_pub, a_msg);
    waitfor(r);
end