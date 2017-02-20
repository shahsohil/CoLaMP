function [z, history] = total_variation_4block(y, lambda, rho, alpha)
% total_variation  Solve a variant of total variation minimization via ADMM
%
% [z, history] = total_variation(y, lambda, rho, alpha)
%
% Solves the following 2x2 block sparse regularized problem via ADMM:
%
%   minimize  (1/2)||z - y||_2^2 + lambda * sum_ij sqrt(z^2_{i,j} +
%   z^2_{i+1,j} + z^2_{i,j+1} + z^2_{i+1,j+1})
%
% where y in R^{nxn}.
%
% The solution is returned in the matrix z.
%
% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.
%
% rho is the augmented Lagrangian parameter.
%
% alpha is the over-relaxation parameter (typical values for alpha are
% between 1.0 and 1.8).

t_start = tic;

% Global constants and defaults

QUIET    = 0;
MAX_ITER = 1000;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;

% Data preprocessing

[n,m] = size(y);
r = mod(m,2);
s = mod(n,2);
% ADMM solver

x1 = zeros(n,m);
x2 = zeros(n,m);
x3 = zeros(n,m);
x4 = zeros(n,m);
z = y;
v1 = zeros(n,m);
v2 = zeros(n,m);
v3 = zeros(n,m);
v4 = zeros(n,m);
c = zeros(n,m);
    
if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
        'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k = 1:MAX_ITER
    
    % x-update
    % update for variable x1
    a = padarray(z - v1,[1 1],'replicate');
    b = (a(2:2:end-1,2:2:end-1).^2 + a(2:2:end-1,3:2:end).^2 + a(3:2:end,2:2:end-1).^2 + a(3:2:end,3:2:end).^2).^0.5;
    c(1:2:end,1:2:end) = b;
    c(1:2:end,2:2:end) = b(:,1:end-r);
    c(2:2:end,1:2:end) = b(1:end-s,:);
    c(2:2:end,2:2:end) = b(1:end-s,1:end-r);
    a = a(2:end-1,2:end-1);
    x1 = a - (a./(rho*c));
    x1(c <= (1/rho)) = 0;
    
    % update for variable x2
    a = padarray(z - v2,[1 1],'replicate');
    b = (a(2:2:end-1,1:2:end-1).^2 + a(2:2:end-1,2:2:end).^2 + a(3:2:end,1:2:end-1).^2 + a(3:2:end,2:2:end).^2).^0.5;
    c(1:2:end,1:2:end) = b(:,1:end-1+r);
    c(1:2:end,2:2:end) = b(:,2:end);
    c(2:2:end,1:2:end) = b(1:end-s,1:end-1+r);
    c(2:2:end,2:2:end) = b(1:end-s,2:end);
    a = a(2:end-1,2:end-1);
    x2 = a - (a./(rho*c));
    x2(c <= (1/rho)) = 0;
    
    % update for variable x3
    a = padarray(z - v3,[1 1],'replicate');
    b = (a(1:2:end-1,2:2:end-1).^2 + a(1:2:end-1,3:2:end).^2 + a(2:2:end,2:2:end-1).^2 + a(2:2:end,3:2:end).^2).^0.5;
    c(1:2:end,1:2:end) = b(1:end-1+s,:);
    c(1:2:end,2:2:end) = b(1:end-1+s,1:end-r);
    c(2:2:end,1:2:end) = b(2:end,:);
    c(2:2:end,2:2:end) = b(2:end,1:end-r);
    a = a(2:end-1,2:end-1);
    x3 = a - (a./(rho*c));
    x3(c <= (1/rho)) = 0;
    
    % update for variable x4
    a = padarray(z - v4,[1 1],'replicate');
    b = (a(1:2:end-1,1:2:end-1).^2 + a(1:2:end-1,2:2:end).^2 + a(2:2:end,1:2:end-1).^2 + a(2:2:end,2:2:end).^2).^0.5;
    c(1:2:end,1:2:end) = b(1:end-1+s,1:end-1+r);
    c(1:2:end,2:2:end) = b(1:end-1+s,2:end);
    c(2:2:end,1:2:end) = b(2:end,1:end-1+r);
    c(2:2:end,2:2:end) = b(2:end,2:end);
    a = a(2:end-1,2:end-1);
    x4 = a - (a./(rho*c));
    x4(c <= (1/rho)) = 0;
    
    % z-update with relaxation
    zold = z;
    
    x1_hat = alpha*x1 +(1-alpha)*zold;
    x2_hat = alpha*x2 +(1-alpha)*zold;
    x3_hat = alpha*x3 +(1-alpha)*zold;
    x4_hat = alpha*x4 +(1-alpha)*zold;
    z = (rho/(4*rho + lambda))*(x1_hat + x2_hat + x3_hat + x4_hat + v1 + v2 + v3 + v4 + (lambda*y/rho));
    
    % v-update
    v1 = v1 + x1 - z;
    v2 = v2 + x2 - z;
    v3 = v3 + x3 - z;
    v4 = v4 + x4 - z;
    
    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(y, lambda, x1, x2, x3, x4, z);
    
    history.r_norm(k)  = sqrt(norm(x1 - z,'fro')^2 + norm(x2 - z,'fro')^2 + norm(x3 - z,'fro')^2 + norm(x4 - z,'fro')^2);
    history.s_norm(k)  = 2*norm(-rho*(z - zold),'fro');
    
    history.eps_pri(k) = sqrt(4*n*m)*ABSTOL + RELTOL*max(norm([x1;x2;x3;x4],'fro'), 2*norm(-z));
    history.eps_dual(k)= sqrt(4*n*m)*ABSTOL + RELTOL*norm(rho*[v1;v2;v3;v4],'fro');
    
    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end
    
    if (k> 50 && history.r_norm(k) < history.eps_pri(k) && ...
            history.s_norm(k) < history.eps_dual(k))
        break;
    end
end

if ~QUIET
    toc(t_start);
end
end

function obj = objective(y, lambda, x1, x2, x3, x4, z)
obj = .5*lambda*norm(z - y,'fro')^2;
a = padarray(x1.^2,[1 1]);
obj =  obj + sum(sum((a(2:2:end-1,2:2:end-1) + a(2:2:end-1,3:2:end) + a(3:2:end,2:2:end-1) + a(3:2:end,3:2:end)).^0.5));
a = padarray(x2.^2,[1 1]);
obj =  obj + sum(sum((a(2:2:end-1,1:2:end-1) + a(2:2:end-1,2:2:end) + a(3:2:end,1:2:end-1) + a(3:2:end,2:2:end)).^0.5));
a = padarray(x3.^2,[1 1]);
obj =  obj + sum(sum((a(1:2:end-1,2:2:end-1) + a(1:2:end-1,3:2:end) + a(2:2:end,2:2:end-1) + a(2:2:end,3:2:end)).^0.5));
a = padarray(x4.^2,[1 1]);
obj =  obj + sum(sum((a(1:2:end-1,1:2:end-1) + a(1:2:end-1,2:2:end) + a(2:2:end,1:2:end-1) + a(2:2:end,2:2:end)).^0.5));
end