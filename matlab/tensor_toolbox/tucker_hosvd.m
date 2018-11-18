function res  = tucker_hosvd(X,R)
%TUCKER_HOSVD Summary of this function goes here
%   Detailed explanation goes here

% N is 3
N = ndims(X); 
% U has 3 room for 3 compositions
U = cell(N,1);
% get 3 part 
for n = 1:3
    U{n} = nvecs(X,n,R(n));
end
core = ttm(X,U,'t');
res = ttensor(core,U);

end

