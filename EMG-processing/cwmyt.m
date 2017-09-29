[emg,activation] = loadData();
param=loadParams();
window = 100;
dim = length(emg);

cc = cwt(emg(1,:), 'amor', 600);
sc =abs(cc).^2;
res = meanfreq(sc); 
res_filt = lpfilter(res', 50, 600);
j=1;
in = nan(2,floor(dim/window));
out = nan(1,floor(dim/window));

for i=window:window:dim
    range = i-window+1:i;
    sample = res_filt(range); 
    in(1,j) = mean(sample(1,:));
    in(2,j) = mean(sample(2,:));
    out(1,j) = mean(activation(range));
    j=j+1;
end

