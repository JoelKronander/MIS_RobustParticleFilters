function [ state, x, w, a, ESS] = DMSFilterALG2( model, y, Np, Nl, ESS_thres,resampletype)
%Simple particle filter proposing samples from two proposals
% N_p samples from p(x_t | x_t-1) 
% ans N_h samples from p( yt | xt),
%
% All the references to equations within this code are with respect to the 
% paper,
% 
% Joel Kronander, Thomas Sch?n, Robust Auxillary Particle Filters using Multiple 
% Importance Sampling. IEEE Statistical Signal Processing Workshop, Jul. 2014.
%
% If you use this code for academic work, please reference the above paper. 
%
% Multiple importance weights according to algorithm 1 in paper
%
% INPUT
%   @model : model struct,
%   @y : measurements, Tx1 Matrix
%   @Np : mumber of particles to sample from prior
%   @Nl : number of particles to sample from likelihood
%   @ESS_thres : Resampling threshold in ESS 
%                if not set resampling is performed at each iteration
%   @resampletype : 0 - Mulitnomial
%                   1 - Systematic
%                   2 - Stratified
%
% OUTPUT
%   @state : state estimates E[x | y_1..T]
%   @x : filter particles %{T,1} Cell array of [nx,N] mats
%   @w : filter weights %{T,} Cell array of [1,N] mats
%   @a  : ancestor indicies %{T,1} Cell array of [1,N] mats
%   @ESS : computed ESS measure over particles
%
% Joel Kronander 2014
% joel.kronander@liu.se

if(nargin < 4)
    ESS_thres = inf;
    resampletype = 1;
end

N = Np + Nl;
T = length(y);
x = cell(T,1); %particles %Tx1 Cell array of nx,N mats
w = cell(T,1); %weights %Tx1 Cell array of 1,N mats
a = cell(T,1); %particle ancestor %Tx1 Cell array of 1,N mats
state = zeros(model.nx,T);
ESS = zeros(T,1); 

Ip = 1:Np; 
Il = Np+1:Np+Nl;

%Propose particles at t=1 from exact initial distribution
x{1} = model.sampP0(N);
w{1} = 1/N*ones(N,1);
ESS(1) = N;

for t = 2:T
    
        if(ESS(t-1) < ESS_thres) % sample ancestor/auxillary index (resample)
            a{t} = resampling(w{t-1}, resampletype);
            w{t} = 1/N*ones(N,1);
        else
            a{t} = 1:N;
            w{t} = w{t-1};
        end
        x_tm1 = x{t-1}(:,a{t});
        
        %Propose new samples from the two proposals
        mixt = (1:2:N);
        nomixt = (2:2:N);
        xtm = model.sampf(x_tm1(:,mixt), t);
        xtnm = model.sampxh(y(:,t), Nl, t);
        x{t}(:,mixt) = xtm;
        x{t}(:,nomixt) = xtnm;
        
        w_h = model.evalh(y(:,t), xtm, t);
        w_p = model.evalf(xtnm, x_tm1(:,nomixt), t);
        
        %Calculate log weights
        w{t}(mixt) = log(w{t}(mixt)) + log( w_h );
        w{t}(nomixt) = log(w{t}(nomixt)) + log( w_p );
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

