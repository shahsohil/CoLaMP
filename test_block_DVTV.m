% Author: Sohil Shah
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Shunsuke Ono (ono@sp.ce.titech.ac.jp)
% Last version: Feb 26, 2014
% Article: S. Ono and I. Yamada, "Deccorelated Vectorial Total Variation," IEEE CVPR 2014
%
% Test images: Berkeley Segmentation Dataset (center region of size 256x256)
%              https://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
addpath subfunctions
addpath Data/images

%%%%%%%%%%%%%%%%%%%%% User Settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%
imname = 'womans.png';
u_org = double(imread(imname))/255;
[n1,n2,n3] = size(u_org);
N = n1*n2*n3;

w1 = 0.5; % weight of luminance variation

%---------------------------------------------------------------
% comment out one 'problemtype' and the corresponding parameters

sigma = 0.1; % noise standard deviation (normalized)
tau = 0.95; % fidelity parameter

%---------------------------------------------------------------

stopcri = 1e-2; % stopping criterion
maxiter = 500; % maximum number of iteration
gamma1 = 0.003; % parameter of PDS
gamma2 = 1/(12*0.01); % parameter of PDS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% observation generation
P = @(z) z;
Pt = @(z) z;
u_obsv = u_org + sigma*randn(size(u_org));
epsilon = tau*sqrt((sigma^2)*N);

psnrInput = EvalImgQuality(u_obsv, u_org, 'PSNR');
disp(['Input PSNR = ', num2str(psnrInput)]);
deltaEInput = EvalImgQuality(u_obsv, u_org, 'Delta2000');
disp(['Input deltaE = ', num2str(deltaEInput)]);

%% definitions for algorithm

% difference operators
D = @(z) cat(4, z([2:n1, n1],:,:) - z, z(:,[2:n2, n2],:)-z);
Dt = @(z) [-z(1,:,:,1); - z(2:n1-1,:,:,1) + z(1:n1-2,:,:,1); z(n1-1,:,:,1)] ...
    +[-z(:,1,:,2), - z(:,2:n2-1,:,2) + z(:,1:n2-2,:,2), z(:,n2-1,:,2)];

% color transform
C = @(z) cat(3, sum(z,3)*(sqrt(3)^(-1)), (z(:,:,1) - z(:,:,3))*(sqrt(2)^(-1)),...
    (z(:,:,1) - 2*z(:,:,2) + z(:,:,3))*(sqrt(6)^(-1)));
Ct = @(z) cat(3, (1/sqrt(3))*z(:,:,1) + (1/sqrt(6))*z(:,:,3) + (1/sqrt(2))*z(:,:,2),...
    (1/sqrt(3))*z(:,:,1) - (2/sqrt(6))*z(:,:,3), (1/sqrt(3))*z(:,:,1) + (1/sqrt(6))*z(:,:,3) - (1/sqrt(2))*z(:,:,2));

% L, prox_f1, and prox_f2
x{1} = u_obsv;
L = @(z) {D(C(z{1})), D(C(z{1})), D(C(z{1})), D(C(z{1})), P(z{1})};
Lt = @(z) {Ct(Dt(z{1})) + Ct(Dt(z{2})) + Ct(Dt(z{3})) + Ct(Dt(z{4})) + Pt(z{5})};

prox_f1{1} = @(z, gamma) ProjDynamicRangeConstraint(z, [0,1]);
prox_f2{1} = @(z, gamma) ProxDVTVnorm(z, gamma, w1,1,1);
prox_f2{2} = @(z, gamma) ProxDVTVnorm(z, gamma, w1,1,2);
prox_f2{3} = @(z, gamma) ProxDVTVnorm(z, gamma, w1,2,1);
prox_f2{4} = @(z, gamma) ProxDVTVnorm(z, gamma, w1,2,2);
prox_f2{5} = @(z, gamma) ProjL2ball(z, u_obsv, epsilon);

y = L(x);
xnum = numel(x);
ynum = numel(y);

%% main loop
for i = 1:maxiter
    % primal update
    xpre = x;
    x = cellfun(@(z1, z2) z1 - gamma1*z2, x, Lt(y), 'UniformOutput', false);
    for j = 1:xnum
        x{j} = prox_f1{j}(x{j}, gamma1);
    end
    
    % dual update
    Ltemp = L(cellfun(@(z1,z2) 2*z1 - z2, x, xpre, 'UniformOutput', false));
    for j = 1:ynum
        Ltemp{j} = y{j} + gamma2 * Ltemp{j};
        y{j} = Ltemp{j} - gamma2 * prox_f2{j}(Ltemp{j}/gamma2, 1/gamma2);
    end
    
    error = sqrt(sum(sum(sum((xpre{1} - x{1}).^2))));
    if error < stopcri
        break;
    end
end
u_res = x{1}; % resulting image

%% result plot

psnrInput = EvalImgQuality(u_res, u_org, 'PSNR');
disp(['Output PSNR = ', num2str(psnrInput)]);
deltaEInput = EvalImgQuality(u_res, u_org, 'Delta2000');
disp(['Output deltaE = ', num2str(deltaEInput)]);

plotsize = [1, 3];
ImgPlot(u_org, 'Original', 1, [plotsize,1]);
ImgPlot(u_obsv, 'Observation', 1, [plotsize,2]);
ImgPlot(u_res, 'Restored', 1, [plotsize,3]);
