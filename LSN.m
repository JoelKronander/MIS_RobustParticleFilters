function [ ssm, data ] = LSN(A, C, Q, R, P0, T)
% Simulates 1D linear system with additive student-t noise;
% x_{t+1} | x_t ~ A*x + t(Q)
% y_t | x_t ~ C*x + t(R)
% p(x_0) = N(0,P0);
% output struct m model
%             struct d data

% Set model params
ssm.linear = 1;
ssm.optprop = 0;
ssm.nx = size(C,2);   % State dimension
ssm.ny = size(C,1);   % Measurement dimension
ssm.A = A;
ssm.C = C;
ssm.Q = Q;
ssm.R = R;
ssm.P0 = P0;
ssm.X0 = zeros(ssm.nx,1);

% Dynamics
ssm.evalf = @(xt1, xt, t) tpdf(xt1-A*xt,Q)'; % evaluates p(x_{t+1} | x_t)
ssm.sampf  = @(xt,t) A*xt + trnd(Q,1,size(xt,2));     % returns N x_{t+1} ~ p(x_{t+1} | x_t)

ssm.evalh = @(yt,xt,t) tpdf(yt-C*xt,R)';   % evaluates p(y_t | x_t)
ssm.sampxh = @(yt, N,t) yt/C + trnd(R, 1, N)/C;
ssm.sampyh = @(xt,t) C*xt + trnd(R,1,size(xt,2));    % returns N y_t ~ p(y_t | x_t)


% Initial state
ssm.sampP0 = @(N) (mvnrnd(ssm.X0,(ssm.P0),N))'; %initital (gaussian) variance

% Simulate System
data.t = (1:T);
x = zeros(ssm.nx,T);
y = zeros(ssm.ny,T);

x(:,1) = mvnrnd(ssm.X0,(ssm.P0)); %samp P(X0)
y(:,1) = ssm.sampyh(x(:,1));
for(t = 2:T)
    x(:,t) = ssm.sampf(x(:,t-1));
    y(:,t) = ssm.sampyh(x(:,t));
end

data.x = x;
data.y = y;

end
