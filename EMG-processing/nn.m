% [emg,activation] = loadData();
param=loadParams();
window = 50;
dim = length(emg);
j=1;
in = nan(1,floor(dim/window));
out = nan(1,floor(dim/window));

for i=window:window:dim
    range = i-window+1:i;
    sample = emg(:,range); 
    in(1,j) = mean(sample(1,:));
    in(2,j) = mean(sample(2,:));
%     if mean(activation(range))<0.5
%         out(j) = 0;
%     else
%         out(j) = 1;
%     end
    out(j) = activation(range(end));
    j=j+1;
end

% figure()
clf
plot(in(1,:));
hold on;
% plot(out);
tmp = in(2,:);
for a=0:0.01:1
    
%     plot(a, mean(tmp(abs(out-a)<0.001)), '*');
    hold on;
%     figure()
%     plot(tmp(abs(output-a)<0.01));
end

% del = out<0.05;
% out(del) = [];
% in(:,del) = [];
% plot(output);
net = feedforwardnet([10,10]);%, 'trainbfg');
% net.layers{1}.transferFcn = 'logsig';
% net.layers{2}.transferFcn = 'logsig';
% net.layers{end}.transferFcn = 'logsig';
[net, p] = train(net, in, out);

clf
load('data/andrew10.mat');

data.robot.biceps.force = (data.robot.biceps.force - 38)/(100-38);
del = data.robot.biceps.force<0.05;
data.subject.triceps.EMG(del) = [];
data.subject.biceps.EMG(del) = [];
data.robot.biceps.force(del) = [];
dim = length(data.robot.biceps.force);

for i=window:window:dim
    range = i-window+1:i;
    testsample(1) = mean(data.subject.biceps.EMG(range)); 
    testsample(2) = mean(data.subject.triceps.EMG(range)); 
    target = mean(data.robot.biceps.force(range));
    plot(i, abs(target - net(testsample')), 'o');
    hold on;
end

