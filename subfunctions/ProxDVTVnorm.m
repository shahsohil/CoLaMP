% Author: Sohil Shah
% Author of original code: Shunsuke Ono (ono@sp.ce.titech.ac.jp)

function[Du] = ProxDVTVnorm(Du, gamma, wlumi, i,j)

[v, h, e, d] = size(Du);
onemat = ones(v, h);

bdim = 2;
f = ones(bdim,bdim);

a = sum(Du(:,:,1,:).^2, 4);
b = imfilter(a,f,'replicate','full').^0.5;
b = b(j:bdim:end,i:bdim:end,:);
c = replicateblock(b,bdim);
c = c(bdim-j+1:bdim-j+v,bdim-i+1:bdim-i+h,:);

threshL = (c.^(-1))*gamma*wlumi/6;

a = sum(sum(Du(:,:,2:3,:).^2, 4),3);
b = imfilter(a,f,'replicate','full').^0.5;
b = b(j:bdim:end,i:bdim:end,:);
c = replicateblock(b,bdim);
c = c(bdim-j+1:bdim-j+v,bdim-i+1:bdim-i+h,:);

threshC = (c.^(-1))*gamma/6;

threshL(threshL > 1) = 1;
threshC(threshC > 1) = 1;
coefL = (onemat - threshL);
coefC = (onemat - threshC);

for l = 1:d
    Du(:,:,1,l) = coefL.*Du(:,:,1,l);
    Du(:,:,2,l) = coefC.*Du(:,:,2,l);
    Du(:,:,3,l) = coefC.*Du(:,:,3,l);
end
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









