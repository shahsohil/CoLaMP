%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A test wrapper for robust compressive image recovery
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preparing background subtracted image using Walking dataset
load('./Data/walkingdata.mat')
b1 = M(:,:,1);
b2 = M(:,:,285);
clear M
x = single(abs(b1-b2));

% finding the threshold value for 97% of energy
e = [];
for i = 1:20
 e(i) = sum(sum(x(x>=i).^2));
end
i = find(e/e(1) < 0.97,1);
K = sum(sum(x>=i));

M = floor(3*K);
MAX_ITER = 10;

lambda = 0.5;
rho = lambda/4;
alpha = 1;
updateLambda = false;

% creating measurement matrix
phi = randn(M,numel(x))/M;
phi = spdiags(1./sqrt(sum(phi.^2,2)),0,M,M)*phi;

%creating noisy observations
y = phi*x(:);
xold = zeros(size(x));
xnew = matching_pursuit(y,phi,xold,K,MAX_ITER,lambda,rho,alpha, updateLambda);

% printing normalized colamp error
colamp_err = norm(xnew-x(:))/norm(x(:));
fprintf('The normalized error for the recovered image is %.3f\n',colamp_err);