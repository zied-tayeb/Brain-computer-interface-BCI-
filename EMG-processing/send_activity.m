if not(robotics.ros.internal.Global.isNodeActive)
    rosinit('localhost');
end
Fs = 600;
window = 150;
mother = 'db2';
level = 4;
x = [-0.3229    0.9636   -0.0389    0.1322   -0.4394];%[-0.0204    0.2127   -0.0461];

sample=nan(150,1);
param.RMSwindow = 50;
topic = strcat('/emg');
emg_sub = rossubscriber(topic);

a_pub = rospublisher('/activation','std_msgs/Float64');
msg = rosmessage(a_pub);

while(1)
    for i=1:150
        sample(i) = emg_sub.LatestMessage.Data;
    end
    
%     sample = lpfilter(sample, 10, Fs/window);

    [C , L] = wavedec(sample, level , mother);
    d4 = detcoef(C,L,level);
    input = mean(abs(d4));
    inputs2 = 0;
    msg.Data = x(1)* input.^2 + x(2) * input + x(3) + x(4)* inputs2.^2 + x(5) * inputs2;

%     msg.Data = x(1)* input^2 + x(2) * input + x(3);
%     msg.Data = sim(net, input');
    
%     RMS = nan(size(sample));
   
%     for k = param.RMSwindow:length(sample)-param.RMSwindow-1
%         RMS(k,:) = rms(sample(k-param.RMSwindow+1:k+param.RMSwindow));
%     end
%     
%     del = isnan(RMS);
%     RMS(del) = [];
%     
%     msg.Data = sim(net, RMS);
    
    send(a_pub, msg);
    i=1;
end