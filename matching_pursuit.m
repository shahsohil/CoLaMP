function [xold] = matching_pursuit(y,phi,xold,K,MAX_ITER,lambda,rho,alpha, updateLambda)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [xold] = matching_pursuit(y,phi,xold,K,MAX_ITER,lambda,rho,alpha)
% This executes Algorithm 2 from the paper.
% Inputs:
% y:            Measured signal
% phi:          Measurement matrix
% xold:         Previously recovered signal (for warm start)
% K:            Sparsity of original signal
% lambda:       weighing parameter
% rho, alpha:   ADMM parameters
% updateLambda: Whether to update lambda after every interval
% Output:
% x:            Recovered signal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tol = 1e-3; % for pcg
[m,n] = size(xold);
xold = xold(:);

for iter = 1:MAX_ITER
    % Step 1
    r = y - phi*xold;
    
    % Step 2
    xnew = phi' * r + xold;
    
    % Step 3
    % In order to test for 8-connected links use the second wrapper    
    [z, ~] = total_variation_4block(reshape(xnew,m,n), lambda, rho, alpha);    
    % [z, ~] = total_variation_cross_links(reshape(xnew,m,n), lambda, rho, alpha);    

    % Step 4
    z = z(:);
    s = (abs(z) > 0.2);
    
    % conjugate gradient step
    A = phi(:,s);
    b = double(A'*y);    
    A = sparse(A'*A);
    t = zeros(m*n,1);
    [t(s==1),~] = pcg(A,b,tol);
    [st,idx] = sort(t,'descend');
    xold = zeros(m*n,1);
    xold(idx(1:K)) = st(1:K);

    if(updateLambda)
        lambda = lambda * (98/100);
        rho = lambda/2;
    end
end

end