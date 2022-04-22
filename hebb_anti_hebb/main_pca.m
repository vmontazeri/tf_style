clear
close all
clc


X = randn(1500, 1);
c = randn(1500, 1);
w = randn(2,2);
LR = 1e-4;
for i = 1 : 20
    
     [w, c, p , s] = recurrent_pca_trainer( X, c, w, LR, 10, 1 );
    pause(.1)
    
end