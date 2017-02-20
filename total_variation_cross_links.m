function [z, history] = total_variation_cross_links(y, lambda, rho, alpha)
% total_variation  Solve a variant of total variation minimization via ADMM
%
% [z, history] = total_variation(y, lambda, rho, alpha)
%
% Solves the following 8-connected block regularized problem via ADMM:
%
%   minimize  (1/2)||z - y||_2^2 + lambda * sum_ij sqrt(z^2_{i,j} +
%   z^2_{i+1,j}) + sqrt(z^2_{i,j} + z^2_{i,j+1}) + sqrt(z^2_{i,j} + z^2_{i+1,j+1})
%    + sqrt(z^2_{i,j} + z^2_{i-1,j+1})
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
%

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
x5 = zeros(n,m);
x6 = zeros(n,m);
x7 = zeros(n,m);
x8 = zeros(n,m);
z = zeros(n,m);
v1 = zeros(n,m);
v2 = zeros(n,m);
v3 = zeros(n,m);
v4 = zeros(n,m);
v5 = zeros(n,m);
v6 = zeros(n,m);
v7 = zeros(n,m);
v8 = zeros(n,m);
c = zeros(n,m);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
        'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k = 1:MAX_ITER
    
    % x-update
    % update for variable x1
    a = padarray(z - v1,[1 1],'replicate');
    b = (a(2:end-1,2:2:end-1).^2 + a(2:end-1,3:2:end).^2).^0.5;    
    c(:,1:2:end) = b;
    c(:,2:2:end) = b(:,1:end-r);
    a = a(2:end-1,2:end-1);
    x1 = a - (a./(rho*c));
    x1(c <= (1/rho)) = 0;
    
    % update for variable x2
    a = padarray(z - v2,[1 1],'replicate');
    b = (a(2:end-1,1:2:end-1).^2 + a(2:end-1,2:2:end).^2).^0.5;
    c(:,1:2:end) = b(:,1:end-1+r);
    c(:,2:2:end) = b(:,2:end);
    a = a(2:end-1,2:end-1);
    x2 = a - (a./(rho*c));
    x2(c <= (1/rho)) = 0;
    
    % update for variable x3
    a = padarray(z - v3,[1 1],'replicate');
    b = (a(2:2:end-1,2:end-1).^2 + a(3:2:end,2:end-1).^2).^0.5;
    c(1:2:end,:) = b;
    c(2:2:end,:) = b(1:end-s,:);
    a = a(2:end-1,2:end-1);
    x3 = a - (a./(rho*c));
    x3(c <= (1/rho)) = 0;
    
    % update for variable x4
    a = padarray(z - v4,[1 1],'replicate');
    b = (a(1:2:end-1,2:end-1).^2 + a(2:2:end,2:end-1).^2).^0.5;
    c(1:2:end,:) = b(1:end-1+s,:);
    c(2:2:end,:) = b(2:end,:);
    a = a(2:end-1,2:end-1);
    x4 = a - (a./(rho*c));
    x4(c <= (1/rho)) = 0;
    
    % update for variable x5
    a = padarray(z - v5,[1 1],'replicate');
    b = (a(2:2:end-1,1:end-1).^2 + a(3:2:end,2:end).^2).^0.5;
    c(1:2:end,:) = b(:,2:end);
    c(2:2:end,:) = b(1:end-s,1:end-1);
    a = a(2:end-1,2:end-1);
    x5 = a - (a./(rho*c));
    x5(c <= (1/rho)) = 0;
    
    % update for variable x6
    a = padarray(z - v6,[1 1],'replicate');
    b = (a(1:2:end-1,1:end-1).^2 + a(2:2:end,2:end).^2).^0.5;
    c(1:2:end,:) = b(1:end-1+s,1:end-1);
    c(2:2:end,:) = b(2:end,2:end);
    a = a(2:end-1,2:end-1);
    x6 = a - (a./(rho*c));
    x6(c <= (1/rho)) = 0;
    
    % update for variable x7
    a = padarray(z - v7,[1 1],'replicate');
    b = (a(2:2:end-1,2:end).^2 + a(3:2:end,1:end-1).^2).^0.5;
    c(1:2:end,:) = b(:,1:end-1);
    c(2:2:end,:) = b(1:end-s,2:end);
    a = a(2:end-1,2:end-1);
    x7 = a - (a./(rho*c));
    x7(c <= (1/rho)) = 0;
    
    % update for variable x8
    a = padarray(z - v8,[1 1],'replicate');
    b = (a(1:2:end-1,2:end).^2 + a(2:2:end,1:end-1).^2).^0.5;
    c(1:2:end,:) = b(1:end-1+s,2:end);
    c(2:2:end,:) = b(2:end,1:end-1);
    a = a(2:end-1,2:end-1);
    x8 = a - (a./(rho*c));
    x8(c <= (1/rho)) = 0;
    
    % z-update with relaxation
    zold = z;
    
    x1_hat = alpha*x1 +(1-alpha)*zold;
    x2_hat = alpha*x2 +(1-alpha)*zold;
    x3_hat = alpha*x3 +(1-alpha)*zold;
    x4_hat = alpha*x4 +(1-alpha)*zold;
    x5_hat = alpha*x5 +(1-alpha)*zold;
    x6_hat = alpha*x6 +(1-alpha)*zold;
    x7_hat = alpha*x7 +(1-alpha)*zold;
    x8_hat = alpha*x8 +(1-alpha)*zold;
    z = (rho/(8*rho + lambda))*(x1_hat + x2_hat + x3_hat + x4_hat + x5_hat + x6_hat + x7_hat + x8_hat + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + (lambda*y/rho));
    
    % v-update
    v1 = v1 + x1 - z;
    v2 = v2 + x2 - z;
    v3 = v3 + x3 - z;
    v4 = v4 + x4 - z;
    v5 = v5 + x5 - z;
    v6 = v6 + x6 - z;
    v7 = v7 + x7 - z;
    v8 = v8 + x8 - z;
    
    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(y, lambda, x1, x2, x3, x4, x5, x6, x7, x8, z);
    
    history.r_norm(k)  = norm([x1;x2;x3;x4;x5;x6;x7;x8] - repmat(z,8,1),'fro');
    history.s_norm(k)  = sqrt(8)*norm(-rho*(z - zold),'fro');
    
    history.eps_pri(k) = sqrt(8*n*m)*ABSTOL + RELTOL*max(norm([x1;x2;x3;x4;x5;x6;x7;x8],'fro'), sqrt(8)*norm(-z));
    history.eps_dual(k)= sqrt(8*n*m)*ABSTOL + RELTOL*norm(rho*[v1;v2;v3;v4;v5;v6;v7;v8],'fro');
    
    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end
    
    if (k>50 && history.r_norm(k) < history.eps_pri(k) && ...
            history.s_norm(k) < history.eps_dual(k))
        break;
    end
end

if ~QUIET
    toc(t_start);
end
end

function obj = objective(y, lambda, x1, x2, x3, x4, x5, x6, x7, x8, z)
obj = .5*lambda*norm(z - y,'fro')^2;
a = padarray(x1.^2,[1 1]);
obj =  obj + sum(sum((a(2:end-1,2:2:end-1) + a(2:end-1,3:2:end)).^0.5));
a = padarray(x2.^2,[1 1]);
obj =  obj + sum(sum((a(2:end-1,1:2:end-1) + a(2:end-1,2:2:end)).^0.5));
a = padarray(x3.^2,[1 1]);
obj =  obj + sum(sum((a(2:2:end-1,2:end-1) + a(3:2:end,2:end-1)).^0.5));
a = padarray(x4.^2,[1 1]);
obj =  obj + sum(sum((a(1:2:end-1,2:end-1) + a(2:2:end,2:end-1)).^0.5));
a = padarray(x5.^2,[1 1]);
obj =  obj + sum(sum((a(2:2:end-1,1:end-1) + a(3:2:end,2:end)).^0.5));
a = padarray(x6.^2,[1 1]);
obj =  obj + sum(sum((a(1:2:end-1,1:end-1) + a(2:2:end,2:end)).^0.5));
a = padarray(x7.^2,[1 1]);
obj =  obj + sum(sum((a(2:2:end-1,2:end) + a(3:2:end,1:end-1)).^0.5));
a = padarray(x8.^2,[1 1]);
obj =  obj + sum(sum((a(1:2:end-1,2:end) + a(2:2:end,1:end-1)).^0.5));
end