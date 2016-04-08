
% Example application of robust particle filters using multiple importance sampling.
%
% All the references to equations within this code are with respect to the 
% paper,
% 
% Joel Kronander, Thomas Sch?n, Robust Auxillary Particle Filters using Multiple 
% Importance Sampling. IEEE Statistical Signal Processing Workshop, Jul. 2014.
%
% If you use this code for academic work, please reference the above paper. 
% 
%
% Written by:
%              Joel kronander (joel.kronander@liu.se)
%              Division of Media and Informtation Technology
%              Link?ping University
%              Last revised on June 30, 2014
%


%% Run comparisons
nExp = 50; %Number of experiments to average

%For storing mse and time between runs
PF_time = zeros(nExp,1); KF_time = PF_time;
LF_time = PF_time; DMSALG1_time = PF_time;
DMSALG2_time = PF_time;

PF_mse = zeros(nExp,1); KF_mse = PF_mse;
LF_mse = PF_mse; DMSALG1_mse = PF_mse;
DMS_mse = PF_mse;

%Parameters
T = 100; %Number of time steps
N = 200; %Number of Particels
N_ess = Inf;%ceil(N/3);
Np = ceil(N/2); %Number of particles to sample from the transition density
Nl = N-Np; %Number of particles to sample prop to the observational density

names = {'PriorFilter', 'LikelihoodFilter', 'DMSFilterBH', 'DMSFilterAUX'};

disp('Running experiments');

for i = 1:nExp

    fprintf(1, 'iteration %i \n', i);
    
    [ssm, data] = LSN(1, 1, 2, 2, 0.01, T);

    % Run Filters
    
    tic
    [statePF, x_f, w_f, a, ESS] = PriorFilter(ssm, data.y, N, N_ess, 3);
    PF_time(i) = toc;   
    PF_mse(i) = MSE(data.x, statePF);
    
    tic
    [stateLF, x_f, w_f, a, ESS] = LikelihoodFilter(ssm, data.y, N, N_ess, 3);
    LF_time(i) = toc;   
    LF_mse(i) = MSE(data.x, stateLF);
          
    tic
    [stateDMSALG1, x_f, w_f, a, ESS] = DMSFilterAlg1(ssm, data.y, Np, Nl, N_ess, 3);
    DMSALG1_time(i) = toc;   
    DMSALG1_mse(i) = MSE(data.x, stateDMSALG1);
    
    tic
    [stateDMSALG2, x_f, w_f, a, ESS] = DMSFilterAlg2(ssm, data.y, Np, Nl, N_ess, 3);
    DMSALG2_time(i) = toc;   
    DMSBHAUX_mse(i) = MSE(data.x, stateDMSALG2);
 
end

% Results
for i = 1:length(names)
    fprintf(' %s',names{i})
end
mse = [PF_mse, LF_mse, DMSALG1_mse, DMSBHAUX_mse];
time = [PF_time, LF_time, DMSALG1_time, DMSALG2_time];
mmse = mean(mse)
mtime = mean(time)

%%
figure(1)
hold on;
plot(data.x, 'k-');
plot(statePF, 'g--');
plot(stateLF, 'c--');
plot(stateDMSALG1, 'r-');
plot(stateDMSALG2, 'b-');
legend('True state', 'PF sampling from Transition density', 'PF sampling prop to Observational density', 'Algorithm 1', 'Allgorithm 2')