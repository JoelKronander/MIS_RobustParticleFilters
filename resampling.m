function i = resampling(q, type, M)

if(nargin < 3)
    M = length(q);
end

qc = cumsum(q);
qc = qc(:);
qc=qc/qc(end); 

if(type == 1) % -- Multinomial, 'simple' from 'sigsys/resample.m'
    u = rand(M,1);
    [~,ind1]=sort([u;qc]);
    ind2=find(ind1<=M);
    i=ind2'-(0:M-1);
elseif(type == 2) % -- Systematic
    u = ((0:M-1)+rand(1))/M;
    i = zeros(1,M); k = 1;
    for j = 1:M
        while(qc(k)<u(j))
            k = k + 1;
        end
        i(j) = k;
    end
elseif(type == 3) % -- Stratified
    u=([0:M-1]'+(rand(M,1)))/M;
    [~,ind1]=sort([u ; qc]);
    ind2=find(ind1<=M);
    i = ind2'-(0:M-1);
else
    error('No such resampling type');
end