i = 1;
ti = zeros(7,1);
for n = 200:100:700
    T = tensor(rand(n,n,n));
     for j = 1:10
     tic
     %X = tucker_als(T,[10,10,10],'tol',1.0e-15,'maxiters',200);
     res = tucker_hosvd(T,[0.1*n,0.1*n,0.1*n]);
     ti(i) = ti(i)+toc;
     end
    ti(i) = ti(i)/10;
    display(n);
    display(ti(i));
    i = i+1;
end