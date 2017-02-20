image_size = 100;
P = phantom('Modified Shepp-Logan',image_size);
y = single(im2uint8(P));

%%%%%%%%%%%%%%%%%%%%%%%%%%%total_variation%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda = 0.06;
rho = lambda/4;
alpha = 1;

[z1, history] = total_variation(y, lambda, rho, alpha);

% reporting
K = length(history.objval);

h = figure;
subplot(3,1,1);
plot(1:K, history.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);
ylabel('f(x^k) + g(z^k)'); xlabel('iter (k)');

subplot(3,1,2);
semilogy(1:K, max(1e-8, history.r_norm), 'k', ...
    1:K, history.eps_pri, 'k--',  'LineWidth', 2);
ylabel('||r||_2');

subplot(3,1,3);
semilogy(1:K, max(1e-8, history.s_norm), 'k', ...
    1:K, history.eps_dual, 'k--', 'LineWidth', 2);
ylabel('||s||_2'); xlabel('iter (k)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%total_variation_cross_links%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda = 0.16;
rho = lambda/4;
alpha = 1;

[z2, history] = total_variation_cross_links(y, lambda, rho, alpha);

% reporting
K = length(history.objval);

h = figure;
subplot(3,1,1);
plot(1:K, history.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);
ylabel('f(x^k) + g(z^k)'); xlabel('iter (k)');

subplot(3,1,2);
semilogy(1:K, max(1e-8, history.r_norm), 'k', ...
    1:K, history.eps_pri, 'k--',  'LineWidth', 2);
ylabel('||r||_2');

subplot(3,1,3);
semilogy(1:K, max(1e-8, history.s_norm), 'k', ...
    1:K, history.eps_dual, 'k--', 'LineWidth', 2);
ylabel('||s||_2'); xlabel('iter (k)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%total_variation_4block%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda = 0.0139;
rho = lambda/4;
alpha = 1;

[z3, history] = total_variation_4block(y, lambda, rho, alpha);

% reporting
K = length(history.objval);

h = figure;
subplot(3,1,1);
plot(1:K, history.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);
ylabel('f(x^k) + g(z^k)'); xlabel('iter (k)');

subplot(3,1,2);
semilogy(1:K, max(1e-8, history.r_norm), 'k', ...
    1:K, history.eps_pri, 'k--',  'LineWidth', 2);
ylabel('||r||_2');

subplot(3,1,3);
semilogy(1:K, max(1e-8, history.s_norm), 'k', ...
    1:K, history.eps_dual, 'k--', 'LineWidth', 2);
ylabel('||s||_2'); xlabel('iter (k)')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
subplot(1,3,1);
imshow(abs(z1)>1,[])
subplot(1,3,2);
imshow(abs(z2)>1,[])
subplot(1,3,3);
imshow(abs(z3)>1,[])