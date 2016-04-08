function [ state, x, w, a, ESS] = PriorFilter( model, y, N, ESS_thres,resampletype)
%Simple particle filter proposing samples from p(x_t+1 | x_t)
%
% INPUT
%   @model : model struct,
%   @y : measurements, Tx1 Matrix
%   @N : mumber of particles
%   @ESS_thres : Resampling threshold in ESS 
%                if not set resampling is performed at each iteration
%   @resampletype : 0 - Mulitnomial
%                   1 - Systematic
%                   2 - Stratified
%
% OUTPUT
%   @state : state estimates E[x | y_1..T]
%   @ESS : computed ESS measure over particles
%
% Joel Kronander 2014
% joel.kronander@liu.se

if(nargin < 4)
    ESS_thres = inf;
    resampletype = 1;
end

T = length(y);
x = cell(T,1); %particles %Tx1 Cell array of nx,N mats
w = cell(T,1); %weights %Tx1 Cell array of 1,N mats
a = cell(T,1); %particle ancestor %Tx1 Cell array of 1,N mats
state = zeros(model.nx,T);
ESS = zeros(T,1); 

%Propose particles at t=1 from exact initial distribution
x{1} = model.sampP0(N);
w{1} = 1/N*ones(N,1);
 for i = 1:model.nx
       state(i,1) = w{1}'*x{1}(i,:)';
end
ESS(1) = N;

for t = 2:T
    
        if(ESS(t-1) < ESS_thres) %resample
            a{t} = resampling(w{t-1}, resampletype);
            w{t} = 1/N*ones(N,1);
        else
            a{t} = 1:N;
            w{t} = w{t-1};
        end
        
        %Propose new samples
        x{t} = model.sampf(x{t-1}(:,a{t}), t);
        
        %Calculate log weights
        w{t} = log(w{t}) + log(model.evalh(y(:,t), x{t}, t));
        wmax = max(w{t}(:));
        W = exp( w{t}(:) - wmax);
        w{t}(:) =  W ./ sum( W );
        ESS(t) = 1/(sum(w{t}.^2));
        
        %Calculate state estimate E[x]
        for i = 1:model.nx
             state(i,t) = w{t}'*x{t}(i,:)';
        end
end


end

