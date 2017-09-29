function [emg, activation] = loadData(des)
    param=loadParams();
    cd data

    % for q=1:11
        samples.wrestling = {};
        samples.calibration = {};

%         des = [17:20];%[16,17,18];
        include = zeros(21,1);

        emg_muscle = 'biceps';
        force_muscle = 'biceps';

        if (strcmp(force_muscle,'triceps'))
            ranges = 70001:140000;
        else
            ranges = 1000:70000;
        end

        for i=1:length(des)
            include(des(i)) = 1;
        end

%         if(include(1)==1)
%             samples.calibration = [samples.calibration, load('19_emec_calibration.mat')];
%             samples.wrestling = [samples.wrestling, load('19_emec.mat')];
%         end

        if(include(2)==1)
            samples.wrestling = [samples.wrestling, load('18_leonard.mat')];
            samples.calibration = [samples.calibration, load('17_leonard_calibration.mat')];
        end

        if(include(3)==1)
            samples.wrestling = [samples.wrestling, load('17_leonard.mat')];
            samples.calibration = [samples.calibration, load('17_leonard_calibration.mat')];
        end


        if(include(4)==1)
            samples.wrestling = [samples.wrestling,load('16_matthias.mat')];
            samples.calibration = [samples.calibration, load('15_matthias_calibration.mat')];
        end

        if(include(5)==1) 
            samples.wrestling = [samples.wrestling, load('15_matthias.mat')];
            samples.calibration = [samples.calibration, load('15_matthias_calibration.mat')];
        end

        if(include(6)==1)
            samples.wrestling = [samples.wrestling, load('20_juri.mat')];
            samples.calibration = [samples.calibration, load('20_juri_calibration.mat')];
        end

        if(include(7)==1) 
            samples.wrestling = [samples.wrestling, load('21_juri.mat')];
            samples.calibration = [samples.calibration, load('20_juri_calibration.mat')];
        end

%         if(include(8)==1)
%             samples.wrestling = [samples.wrestling, load('22_julian.mat')];
%             samples.calibration = [samples.calibration, load('22_julian_calibration.mat')];
%         end
% 
%         if(include(9)==1)
%             samples.wrestling = [samples.wrestling, load('23_julian.mat')];
%             samples.calibration = [samples.calibration, load('22_julian_calibration.mat')];
%         end
        % 
        if(include(10)==1)
            samples.wrestling = [samples.wrestling, load('25_konstantin.mat')];
            samples.calibration = [samples.calibration, load('24_konstantin_calibration.mat')];
        end
        % 
        if(include(11)==1)
            samples.wrestling = [samples.wrestling, load('26_christoph.mat')];
            samples.calibration = [samples.calibration, load('26_christoph_calibration.mat')];
        end


        if(include(12)==1)
            samples.calibration = [samples.calibration, load('andrew1_calibration.mat')];
            samples.wrestling = [samples.wrestling,  load('andrew1.mat')];
        end

         if(include(13)==1)
            samples.calibration = [samples.calibration, load('andrew1_calibration.mat')];
            samples.wrestling = [samples.wrestling,  load('andrew2.mat')];
         end

        if(include(14)==1)
            samples.calibration = [samples.calibration, load('andrew1_calibration.mat')];
            samples.wrestling = [samples.wrestling,  load('andrew3.mat')];
        end

%         if(include(15)==1)
%             samples.calibration = [samples.calibration, load('andrew1_calibration.mat')];
%             samples.wrestling = [samples.wrestling,  load('andrew4.mat')];
%         end

        if(include(16)==1)
            samples.calibration = [samples.calibration, load('andrew1_calibration.mat')];
            samples.wrestling = [samples.wrestling,  load('andrew5.mat')];
        end

        if(include(17)==1)
            samples.calibration = [samples.calibration, load('andrew1_calibration.mat')];
            samples.wrestling = [samples.wrestling,  load('andrew6.mat')];
        end

        if(include(18)==1)
            samples.calibration = [samples.calibration, load('andrew1_calibration.mat')];
            samples.wrestling = [samples.wrestling,  load('andrew7.mat')];
        end
        
        if(include(19)==1)
            samples.calibration = [samples.calibration, load('andrew1_calibration.mat')];
            samples.wrestling = [samples.wrestling,  load('andrew8.mat')];
        end
        
        if(include(20)==1)
            samples.calibration = [samples.calibration, load('andrew1_calibration.mat')];
            samples.wrestling = [samples.wrestling,  load('andrew9.mat')];
        end
        
        if(include(21)==1)
            samples.calibration = [samples.calibration, load('andrew1_calibration.mat')];
            samples.wrestling = [samples.wrestling,  load('andrew10.mat')];
        end
        
        cd ..
        
        emg = nan(2,65000*8);
        forces = nan(2,65000*8);
        
        d = designfilt('bandpassiir','FilterOrder',4, ...
        'HalfPowerFrequency1',5,'HalfPowerFrequency2',300, ...
        'SampleRate',600);
    
        wo = 50/(600/2);  bw = wo/35;
        [bn,an] = iirnotch(wo,bw);


        for z=1:2
            if z==2
                if strcmp(emg_muscle, 'biceps')
                    emg_muscle = 'triceps';
                else
                    emg_muscle = 'biceps';
                end
            end
            k=1;

            for i=1:length(samples.calibration)
                clf;
                cur = samples.wrestling(i);
                cur_c = samples.calibration(i);

                nfilt_emg = cur{1}.data.subject.(emg_muscle).EMG(ranges);
                nfilt_emg = filter(d,nfilt_emg);
                
                nfilt_emg = filter(bn,an, nfilt_emg);
                
                
%                 RMS = nan(size(nfilt_emg));
%                 for j = param.RMSwindow:length(nfilt_emg)-param.RMSwindow-1
%                     RMS(j,:) = rms(nfilt_emg(j-param.RMSwindow+1:j+param.RMSwindow));
%                 end

            %     RMS = RMS/max(max(RMS));

            %     plot(RMS); hold on; 
            %     mvc(1:length(RMS))=cur_c{1}.calib.calibration.triceps.MVC;
            %     plot(mvc);

            %     n = isnan(RMS);

                len = length(cur{1}.data.subject.(emg_muscle).EMG(ranges));
                
                tmp = nfilt_emg/cur_c{1}.calib.calibration.(emg_muscle).MVC;
%                 tmp = (tmp-min(tmp))/(max(tmp) - min(tmp));
                emg(z,k:len+k-1) = tmp;

                forces(z,k:len+k-1) = cur{1}.data.robot.(force_muscle).force(ranges);
        %         plot(forces(k:len+k-1));
                k = k + len + 1;
            end


        end
    del = isnan(emg(1,:));
    emg(:,del) = [];
    forces(:,del) = [];
    
    forces(forces<38) = 38;
    
    activation = (forces(1,:)-38)/(100-38);
%     activation = forces;
%     for i=1:length(activation)-1500
%         if activation(i)-activation(i+200)<-0.1
%             activation(i+100:i+400) = activation(i+1500);
%         end
%     end
    clf;
    plot(emg(2,:));
    hold on;plot(2*activation);
end

%     clf;
%     
%     del = isnan(emg(1,:));
%     emg(:,del) = [];
%     forces(:,del) = [];
%     
%     forces(forces<38) = 38;
% 
%     plot(1:length(emg),100*emg(1,:))
%     hold on;
%     plot(1:length(forces),forces(1,:))
%     
%     figure()
%     
%     plot(1:length(emg),100*emg(2,:))
%     hold on;
%     plot(1:length(forces),forces(1,:))
%     
%     dim=length(emg);
%     window = 200;
%     overlap = 0;
% 
%     emg_mav = nan(1,floor(dim/(window-overlap))-1);
%     force_mav = nan(1,floor(dim/(window-overlap))-1);
%     emg_rms = nan(1,floor(dim/(window-overlap))-1);
%     inputs = nan(floor(dim/window),1);
%     outputs = nan(floor(dim/window),1);
%     bp =  nan(floor(dim/window),2);
% 
%     j=1;
% 
%     level = 3;
%     mother = 'db2';
% 
%     emg(isnan(emg)) = [];
%     forces(isnan(emg)) = [];
% 
%     clf;
%     for i=window:window:dim
%         clf;
%         range = i-window+1:i;
%         sample = emg(:,range); 
% 
%         bp(j,1) = mean(sample(1,:));
%         bp(j,2) = mean(sample(2,:));
% %         [C , L] = wavedec(sample , level , mother);
% %         cA3 = appcoef(C , L , mother, level);
% %         r = cA3;
% %         rMinMax = minmax(r);
% %         s = sample;
% %         sMinMax = minmax(sample);
% %         ret = (((r - rMinMax(1)) / (rMinMax(2) - rMinMax(1))) * (sMinMax(2) - sMinMax(1))) + sMinMax(1); 
%     %     plot(ret);hold on;
% 
%         force = forces(1,range)';
% 
% %         inputs(j,:) = max(ret);
%         outputs(j,:) = mean(force);
%     %     inputs((j-1)*27 +1 :j*27) = ret;
%     %     outputs((j-1)*27 +1 :j*27) = mean(force);
% 
%     %     subplot(211); plot(sample); 
%     %     title('Original signal'); 
%     %     subplot(212); plot(ret); 
%     %     title('Wavelet decomposition structure, level 3, db2 mother function') 
%     %     xlabel(['Coefs for approx. at level 3 ' ... 
%     %             'and for det. at levels 3, 2 and 1'])
% 
%     %     inputs = [inputs,ret];
%     %     outputs = [outputs,mean(force)];
% 
%         j=j+1;
%     end
%        
%     bp(isnan(bp(1,:)),:) = [];
%     outputs(isnan(bp(1,:)),:) = [];
%     
%     bp(isnan(outputs),:) = [];
%     outputs(isnan(outputs)) = [];
%     
%     mdl = fitlm(bp,outputs,'linear','RobustOpts','on')
%     
% %     clf;
% %     plot(bp,outputs,'o');
% %     % hold on;
% %     % p = polyfit(bp,outputs,7);
% %     % outputs2 = polyval(p,bp);
% %     % plot(bp,outputs2);
% %     % hold off;
% %     clf
% %     f = fit(bp, outputs, 'poly34');
% %     
% %     plot(f,bp, outputs);
% %     % plot(100*emg);hold on;
% %     plot(100*inputs);hold on;
% %     plot(outputs);
% %     clf;
%     % modelfun = @(b, x) b(1)-b(1)./(1+4/b(2).*(cosh(b(3)/b(4).*x)).^2);
%     % modelfun=@(b,x)(1./(1+exp(-b(1).*x-b(2))));
%     % modelfun = @(b,x)x(:,1)/8 + b(1) - b(1)*((1 - (3/(b(1)*4))).^(x(:,1)/2));
%     % modelfun = @(b,x)b(1) + b(2)*x(:,1).^b(3) + b(4)*x(:,2).^b(5);
% 
%     % net = feedforwardnet([10,10], 'traingdx');
%     % net.layers{1}.transferFcn = 'logsig';
%     % net.layers{2}.transferFcn = 'logsig';
%     % [net, p] = train(net, inputs', outputs');
%     % y = net(inputs');
%     % perf = perform(net, y, outputs')
% % end
