function [L,z, Y,x,v,history] = robustpca_l2sparsity_mxm_block(M, mu, lambda,bdim, x,v,Y,z)
% Solve a robust pca problem using L2 sparsity using ADMM
%
% [L,z, Y,history] = robustpca_l2sparsity_4block(M, mu, mu, lambda)
%
% Solves the following problem via ADMM:
%
%   minimize  ||L||_* + lambda * sum_ijk sqrt(z^2_{i,j,k} +
%   z^2_{i+1,j,k} + z^2_{i,j+1,k} + z^2_{i+1,j+1,k})
%   s.t. L + Z = M
%
% where M \in R^{nxmxr} with (n,m) being the frame size and r is no. of frames
%
% The solution L is low rank matrix and z is sparse matrix.
%
% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.
%
% mu is the augmented Lagrangian parameter.
% lambda is the regularization parameter which controls the sparsity.

t_start = tic;

% Global constants and defaults

QUIET    = 0;
MAX_ITER = 1000;
ABSTOL   = 0;
RELTOL   = 1e-7;

[n,m,p] = size(M);
bsize = bdim^2;

% if it is not a warm start
if(nargin < 5)
    x = zeros(n,m,p,bsize);
    v = zeros(n,m,p,bsize);
    z = zeros(n,m,p);
    Y = zeros(n,m,p);
end

h = ones(bdim,bdim);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
        'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

% ADMM solver
for k = 1:MAX_ITER
    
    for i = 1:bdim
        for j = 1:bdim
            idx = (i-1)*bdim + j;
            a = z - v(:,:,:,idx);
            b = imfilter(a.^2,h,'replicate','full').^0.5;
            b = b(j:bdim:end,i:bdim:end,:);
            c = replicateblock(b,bdim);
            c = c(bdim-j+1:bdim-j+n,bdim-i+1:bdim-i+m,:);
            temp = a - (lambda*a./(mu*c));
            temp(c <= (lambda/mu)) = 0;
            x(:,:,:,idx) = temp;
        end
    end
       
    %update for L
    A = reshape(M-z+Y,n*m,p);
    [U,S,V] = svd(A,0); % one can improve the speed by computing the partial svd in the subsequent iterations
    singval = max(S-(1/mu),0);
    L = U*singval*V';
    L = reshape(L,n,m,p);
    
    % z-update with relaxation
    zold = z;
    
    z = (1/(bsize + 1))*(sum(x,4) + sum(v,4) + (M-L+Y));
    
    % v-update
    v = v + x - repmat(z,1,1,1,bsize);
    
    %Y update
    Y = Y + (M-L-z);
    
    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(diag(singval), lambda, x, h, bdim);
    
    history.r_norm(k)  = sqrt(norm4(x-repmat(z,1,1,1,bsize)) + norm3(M-L-z));
    history.s_norm(k)  = mu*sqrt((bsize+1) * norm3(z - zold));
    
    history.eps_pri(k) = sqrt((bsize+1)*n*m*p)*ABSTOL + RELTOL*sqrt(max([norm3(M),norm3(L) + norm4(x),(bsize+1)*norm3(z)]));
    history.eps_dual(k)= sqrt((bsize+1)*n*m*p)*ABSTOL + RELTOL*mu*sqrt(norm4(v) + norm3(Y));
    
    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end
    
    if (history.r_norm(k) < history.eps_pri(k) && ...
            history.s_norm(k) < history.eps_dual(k))
        break;
    end

end

if ~QUIET
    toc(t_start);
end
end

function val = norm3(X)
val=sum(sum(sum(X.^2)));
end

function val = norm4(X)
val=sum(sum(sum(sum(X.^2))));
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

function obj = objective(singval, lambda, x, h, bdim)
obj = sum(singval);
b = imfilter(x.^2,h,'full').^0.5;
for i = 1:bdim
    for j = 1:bdim
        idx = (i-1)*bdim + j;
        obj = obj + lambda*sum(sum(sum(b(j:bdim:end,i:bdim:end,:,idx))));
    end
end
end