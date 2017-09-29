param=loadParams();

[emg,activation] = loadData([12:20]);


for i=1:2
    emg(i,:) = (emg(i,:) - mean(emg(i,:)))/std(emg(i,:));
end;

% emg = emg(2,:);

% plot(emg);hold on;
% plot(activation);

dim=length(emg);
Fs = 599;
window = 150;
overlap = 0;

inputs = nan(floor(dim/window),2);
inputs2 = nan(floor(dim/window),2);
outputs = nan(floor(dim/window),1);


j=1;
 
level = 3;
mother = 'db2';

% subplot(211); plot(sample); 
% title('Original signal'); 
% subplot(212); plot(ret); 
% title('Wavelet decomposition structure, level 3, db2 mother function') 
% xlabel(['Coefs for approx. at level 3 ' ... 
%         'and for det. at levels 3, 2 and 1'])
    
clf;
for i=window:window:dim
    range = i-window+1:i;
    sample = emg(:,range); 
    
    [C1 , L1] = wavedec(sample(1,:) , level , mother);
%     cA3 = appcoef(C , L , mother, level);
    d31 = detcoef(C1,L1,level);
    
    [C2 , L2] = wavedec(sample(2,:) , level , mother);
    d32 = detcoef(C2,L2,level);
%     r = cA3;
%     rMinMax = minmax(r);
%     s = sample;
%     sMinMax = minmax(sample);
%     ret = (((r - rMinMax(1)) / (rMinMax(2) - rMinMax(1))) * (sMinMax(2) - sMinMax(1))) + sMinMax(1); 
%     plot(ret);hold on;
    
    a = activation(range)';
    
    inputs(j,2) = var(abs(d32));%mean(d3);
    inputs2(j,1) = mean(abs(d31));
    inputs(j,1) = mean(abs(d32));
%     inputs(j,2) = std(d3);
    outputs(j) = mean(a);
    
    j=j+1;
end

for i=1:2
    inputs(:,i) = lpfilter(inputs(:,i), 10, Fs/window);
     inputs2(:,i) = lpfilter(inputs(:,i), 10, Fs/window);
end

%% LSQ optimization

inputs = inputs(:,1);
inputs2 = inputs2(:,1);
fun = @(x) x(1)* inputs.^2 + x(2) * inputs + x(3) + x(4)* inputs2.^2 + x(5) * inputs2 - outputs;

options = optimoptions('lsqnonlin','Display','iter');
options.Algorithm = 'levenberg-marquardt';
x0 = [1,1,1,1,1];
[x,resnorm,residual,exitflag,out] = lsqnonlin(fun,x0,[],[],options);


%% NN training
% in = inputs';
% out = outputs';
% net = feedforwardnet([6,6,6]);
% % net.layers{1}.transferFcn = 'poslin';
% % net.layers{2}.transferFcn = 'poslin';
% % net.layers{3}.transferFcn = 'poslin';
% net.trainParam.epochs = 30000;
% ing = nndata2gpu(in);
% outg = nndata2gpu(out);
% net = configure(net,in,out);
% [net, p] = train(net, ing, outg,'useParallel', 'yes', 'useGPU','yes','showResources','yes');


%% testing data prep

[emg,activation] = loadData([21]);
dim=length(emg);
Fs = 599;
window = 150;

inputs = nan(floor(dim/window),2);
inputs2 = nan(floor(dim/window),2);
outputs = nan(floor(dim/window),1);

for i=1:2
    emg(i,:) = (emg(i,:) - mean(emg(i,:)))/std(emg(i,:));
end;

j=1;

for i=window:window:dim
    range = i-window+1:i;
    sample = emg(:,range); 
    
    [C1 , L1] = wavedec(sample(1,:) , level , mother);
%     cA3 = appcoef(C , L , mother, level);
    d31 = detcoef(C1,L1,level);
    
    [C2 , L2] = wavedec(sample(2,:) , level , mother);
    d32 = detcoef(C2,L2,level);
    
    a = activation(range)';
    
    inputs(j,2) = var(abs(d32));%mean(d3);
    inputs2(j,1) = mean(abs(d31));
    inputs(j,1) = mean(abs(d32));
%     inputs(j,2) = std(d3);
    outputs(j) = mean(a);
    
    j=j+1;
end

for i=1:2
   inputs(:,i) = lpfilter(inputs(:,i), 10, Fs/window);
   inputs2(:,i) = lpfilter(inputs(:,i), 10, Fs/window);
end

%% LSQ test

inputs = inputs(:,1);
inputs2 = inputs2(:,1);
test_outputs = x(1)* inputs.^2 + x(2) * inputs + x(3) + x(4)* inputs2.^2 + x(5) * inputs2;
clf
plot(test_outputs);
hold on;
plot(outputs);

%% NN test

% test_outputs = sim(net, inputs');
% clf;
% plot(test_outputs)
% hold on;plot(outputs)



