% Author: Shunsuke Ono (ono@sp.ce.titech.ac.jp)

function [out] = EvalImgQuality(u, u_org, Qtype, varargin)

if ~isempty(varargin)
    dynamic = varargin{1};
else
    dynamic = 1;
end
if strcmp(Qtype, 'PSNR')
    MSE = sum(sum(sum((u - u_org).^2)));
    MSE= MSE/(numel(u));
    out = 10 * log10(dynamic^2/MSE);
elseif strcmp(Qtype, 'Delta2000')
    
    %wp = whitepoint('d65');
    C = makecform('srgb2lab');
    u = u/dynamic;
    u_org = u_org/dynamic;
    [y, x, z] = size(u);
    
    Ild = applycform(u,C);
    Jld = applycform(u_org,C);
    
    Ilds = [reshape(Ild(:,:,1),[y*x 1]) reshape(Ild(:,:,2),[y*x 1])  reshape(Ild(:,:,3),[y*x 1]) ];
    Jlds = [reshape(Jld(:,:,1),[y*x 1]) reshape(Jld(:,:,2),[y*x 1])  reshape(Jld(:,:,3),[y*x 1]) ];
    
    d2k = deltaE2000(Ilds, Jlds);
    out= mean(d2k(:));
end