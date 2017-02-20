%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A test wrapper for robust compressive signal recovery
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Shepp-Logan
image_size = 80;
P = phantom('Modified Shepp-Logan',image_size);
x = zeros(100,100);
x(11:90,11:90) = single(im2uint8(P));

K = sum(sum(x>0));
M = floor(2*K); % M=2K

% SNR can be varied to from 5 dB to 20 dB to generate plot 4(b)
SNR = 10;

MAX_ITER = 10;

% In our implementation we divide the objective by lambda, hence here is it
% defined inversely
lambda = 1/16;

% rho and alpha are ADMM parameters
rho = lambda/2;
alpha = 1;

% creating measurement matrix
phi = randn(M,numel(x))/M;
phi = spdiags(1./sqrt(sum(phi.^2,2)),0,M,M)*phi;

%creating noisy observations
y = phi*x(:);
sigma = rms(y)/10^(SNR/20);
e = sigma*randn(M,1);
y = y + e;

updateLambda = true;
% In case of warm start initialize xold with previously recovered vector
xold = zeros(size(x));
t = tic;
xnew = matching_pursuit(y,phi,xold,K,MAX_ITER,lambda,rho,alpha, updateLambda);
toc(t);
