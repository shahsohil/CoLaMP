function [L,z,history] = robustpca_l2sparsity_mxm_block_splitting(M, mu,lambda,bdim, L,z)
% Solve a robust pca problem using L2 structured sparsity using proximal
% gradient (forward backward splitting).
%
% We use FASTA package to solve this. http://www.cs.umd.edu/~tomg/projects/fasta/
%
% [L,z,history] = robustpca_l2sparsity_mxm_block_splitting(M, mu, lambda,bdim, L,z)
%
% Solves the following problem via prox grad:
%
%   minimize  ||L||_* + lambda * sum_ijk sqrt(z^2_{i,j,k} +
%   z^2_{i+1,j,k} + z^2_{i,j+1,k} + z^2_{i+1,j+1,k}) + mu/2 ||M-L-z||^2_F
%
% where M in R^{nxmxp} where (n,m) is the frame size and p is # of frames
%
% The solution is returned in the matrix L and z. The input L and z can be 
% used for warm start.
%
% history is a structure that contains the objective value, the primal norm
% and the tolerances for the primal norms at each iteration.
%
% lambda and mu are the regularization parameter which controls the
% sparsity and error in constraint.

t_start = tic;

% Global constants and defaults

QUIET    = 0;
MAX_ITER = 100;
ABSTOL   = 0;
RELTOL   = 1e-5;
eps = 10^-6;

% Data preprocessing
[n,m,p] = size(M);
M = M(:);

if(nargin < 5)
    z = zeros(n*m*p,1);
    L = zeros(n*m*p,1);
else
    z = z(:);
    L = L(:);
end

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
        'r norm', 'eps pri', 'objective','match error');
end

%%%%%%%%%%% Fasta functions %%%%%%%%%%
grad = @(X)gradient(X,M,bdim,n,m,p,lambda,eps,mu);
func = @(X)fobjective(X,M,bdim,n,m,p,lambda,eps,mu);
gfunc = @(X)gobjective(X,n,m,p);
proxg = @(X,t)proximal(X,n,m,p,t);

%%% Fasta Parameters %%%
opts = [];
opts.maxIters = 20;  %Set this to something small
opts.stopRule = 'ratioResidual';
opts.tol = 1e-1;
opts.verbose = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k = 1:MAX_ITER
    Lold = L;
    zold = z;    
    
    X = fasta(@(x)x, @(x)x, func, grad, gfunc, proxg, [L;z], opts);    
    L = X(1:n*m*p);
    z = X(n*m*p+1:end);
    
    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(reshape(L,n*m,p),z,func);
    
    history.r_norm(k)  = sqrt(sum(([Lold;zold]-[L;z]).^2));
    
    history.eps_pri(k) = sqrt(n*m*p)*ABSTOL + RELTOL*max([norm(L),norm(z)]);
    
    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.2f\t%2.6f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.objval(k),norm(M-L-z,'fro')/norm(M,'fro'));
    end
    
    if (k>500 && history.r_norm(k) < history.eps_pri(k))
        break;
    end    
end

L = reshape(L,n,m,p);
z = reshape(z,n,m,p);

if ~QUIET
    toc(t_start);
end

end

function obj = objective(L,z,f)
S = svd(L);
obj = sum(S);
obj = obj + f([L(:);z]);
end

function val = replicateblock(x,bdim)
[n,m,p] = size(x);
y = x(:,:,:,ones(1,bdim));
z = reshape(reshape(y,numel(x),bdim)',n*bdim,m,p);
x = permute(z,[2 1 3]);
y = x(:,:,:,ones(1,bdim));
z = reshape(reshape(y,numel(x),bdim)',m*bdim,n*bdim,p);
val = permute(z,[2 1 3]);
end

function [G] = gradient(X,M,bdim,n,m,p,lambda,eps,mu)
L = X(1:n*m*p);
z = reshape(X(n*m*p+1:end),n,m,p);
g = zeros(size(z));
h = ones(bdim,bdim);

b = imfilter(z.^2,h,'replicate','full');
b = (b+ones(size(b))*eps).^0.5;

for i = 1:bdim
    for j = 1:bdim
        d = b(j:bdim:end,i:bdim:end,:);
        c = replicateblock(d,bdim);
        c = c(bdim-j+1:bdim-j+n,bdim-i+1:bdim-i+m,:);
        g = g + (1./c);
    end
end

g = lambda*z.*g;
G = [-mu*(M-L-z(:));g(:) - mu*(M-L-z(:))];
end

function [f] = fobjective(X,M,bdim,n,m,p,lambda,eps,mu)
L = X(1:n*m*p);
z = X(n*m*p+1:end);
a = reshape(z,n,m,p);
h = ones(bdim,bdim);

b = imfilter(a.^2,h,'replicate','full');
b = (b+ones(size(b))*eps).^0.5;
f = lambda*sum(b(:)) + mu*sum((M-L-z).^2)/2;
end

function [g] = gobjective(X,n,m,p)
L = reshape(X(1:n*m*p),n*m,p);
S = svd(L);
g = sum(S);
end

function [G] = proximal(X,n,m,p,t)
L = reshape(X(1:n*m*p),n*m,p);
z = X(n*m*p+1:end);

% one can improve the runtime by computing the partial svd in the subsequent iterations
[U,S,V] = svd(L,0);

singval = max(S-t,0);
L = U*singval*V';
G = [L(:);z];
end
