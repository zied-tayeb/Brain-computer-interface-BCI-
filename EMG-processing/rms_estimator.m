param=loadParams();

[emg,activation] = loadData([17:21]);
% [emg,activation] = loadData([22]);

for i=1:2
    emg(i,:) = (emg(i,:) - mean(emg(i,:)))/std(emg(i,:));
end;

[trainInd,valInd,testInd] = divideblock(length(emg),0.8,0.2,0);
trainemg = emg(:,trainInd);
testemg = emg(:,valInd);

traina = activation(trainInd);
testa = activation(valInd);

dim=length(trainemg);
Fs = 599;
window = 150;
param.RMSwindow = 50;
overlap = 0;

inputs = nan(floor(dim/window),window-2*param.RMSwindow);
% inputs = nan(floor(dim/window),1);
outputs = nan(floor(dim/window),1);


j=1;
z = 1;
clf;
for i=window:window:dim
    range = i-window+1:i;
    sample = trainemg(2,range); 
    sample = sample';
    RMS = nan(size(sample));
   
    for k = param.RMSwindow:length(sample)-param.RMSwindow-1
        RMS(k,:) = rms(sample(k-param.RMSwindow+1:k+param.RMSwindow));
    end
    
    del = isnan(RMS);
    RMS(del) = [];
    
    inputs(z,:) = RMS;
%     inputs(z,2) = std(sample);
%     inputs(z,3) = meanfreq(sample);
    outputs(z) = mean(traina(range));
    
    z = z + 1;
end


for i=1:1
   inputs(:,i) = lpfilter(inputs(:,i), 10, Fs/window);
end

%%
% inputs = inputs(:,1);
% % inputs2 = inputs2(:,1);
% fun = @(x) x(1)* inputs.^2 + x(2) * inputs + x(3) - outputs;
% 
% options = optimoptions('lsqnonlin','Display','iter');
% options.Algorithm = 'levenberg-marquardt';
% x0 = [1,1,1];
% [x,resnorm,residual,exitflag,out] = lsqnonlin(fun,x0,[],[],options);

%%
in = inputs';
out = outputs';
net = feedforwardnet([6,6,6]);
% net.layers{1}.transferFcn = 'poslin';
% net.layers{2}.transferFcn = 'poslin';
% net.layers{3}.transferFcn = 'poslin';
net.trainParam.epochs = 30000;
ing = nndata2gpu(in);
outg = nndata2gpu(out);
net = configure(net,in,out);
[net, p] = train(net, ing, outg,'useParallel', 'yes', 'useGPU','yes','showResources','yes');

%%

dim=length(testemg);
Fs = 599;

inputs = nan(floor(dim/window),window-2*param.RMSwindow);
outputs = nan(floor(dim/window),1);


z=1;

for i=window:window:dim
    range = i-window+1:i;
    sample = testemg(2,range); 
    sample = sample';
    RMS = nan(size(sample));
   
    for k = param.RMSwindow:length(sample)-param.RMSwindow-1
        RMS(k,:) = rms(sample(k-param.RMSwindow+1:k+param.RMSwindow));
    end
    
    del = isnan(RMS);
    RMS(del) = [];
    
    inputs(z,:) = RMS;
%     inputs(z,2) = std(sample);
%     inputs(z,3) = meanfreq(sample);
    outputs(z) = mean(testa(range));
    
    z = z + 1;
    
end


for i=1:1
   inputs(:,i) = lpfilter(inputs(:,i), 10, Fs/window);
end

%% NN test
test_outputs = sim(net, inputs');
test_outputs = test_outputs';
clf;
hold on;
% plot(inputs);
hold on;plot(outputs)
plot(test_outputs);

%%
% inputs = inputs(:,1);
% test_outputs = x(1)* inputs.^2 + x(2) * inputs + x(3);
% 
% clf
% hold on;
% % plot(inputs);
% plot(outputs);
% plot(test_outputs);
%%
%% stats
rmse = sqrt(mean((outputs-test_outputs).^2));
cc = corrcoef(outputs,test_outputs);

sst = sum((outputs-mean(outputs)).^2);
ssr = sum((test_outputs-mean(outputs)).^2);
rsquared = ssr/sst;

%% save eval
% 
% training_duration = length(traina)/600;
% testing_duration = length(testa)/600;
% 
% nn_rms = {};
% nn_rms.data = 17:21;
% nn_rms.train_duration_s = length(traina)/600;
% nn_rms.test_duration_s = length(testa)/600;
% % nn_rms.residual = residual;
% nn_rms.corrcoef = cc;
% nn_rms.rmse = rmse;
% nn_rms.rsquared = rsquared;
% nn_rms.outputs = outputs;
% nn_rms.test_outputs = test_outputs;
% nn_rms.inputs = inputs;
% nn_rms.performance = p;
% 
% save('nn_rms.mat', 'nn_rms'); 

%%
%% net pic
jframe = view(net);

%# create it in a MATLAB figure
hFig = figure('Menubar','none', 'Position',[100 100 565 166]);
jpanel = get(jframe,'ContentPane');
[~,h] = javacomponent(jpanel);
set(h, 'units','normalized', 'position',[0 0 1 1])

%# close java window
jframe.setVisible(false);
jframe.dispose();

%# print to file
set(hFig, 'PaperPositionMode', 'auto')
saveas(hFig, 'out.png')

%# close figure
close(hFig)
