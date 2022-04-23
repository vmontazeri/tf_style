function [w, c, p, s] = recurrent_pca_trainer( X, c, w, LR, n_epochs, verbose )

% generalized Hebbian learning
% Note the elements across the memory and input seq should be correlated
% i.e. element 1 in x and element 1 in c and element 2 in x and element 2
% in c. This is different from Stilp data structure.
% ************ Required **************
% X: input, vector of n_features x 1
% optional
% n_epochs
% verbose

num_features  = size(X,1);

assert(nargin >= 3);
switch nargin
    case 3
        LR = 0.001;
        n_epochs = 1;
        verbose = 0;
    case 4
        n_epochs = 1;
        verbose = 0;
    case 5
        verbose = 0;
end

num_seqs = size(X, 2);

for epoch = 1 : n_epochs
    
    dw = zeros(2,2);
    p = zeros(num_seqs, num_features);
    s = zeros(num_seqs, num_features);
        
    if(epoch>n_epochs/2), LR = LR/2; end    
    
    for seq_num = 1 : num_seqs
        
        x = X(:, seq_num);
        W = x * c';
        z = W * x;
        z = z * ( rms(x)/rms(z) );
        
        in = [c x]';
        p(seq_num,:) = w(1,:) * in;
        s(seq_num,:) = w(2,:) * in;
        temp11 = LR * p(seq_num,:) * ( in(1,:) - w(1,1) * p(seq_num,:) )';
        temp12 = LR * p(seq_num,:) * ( in(2,:) - w(1,2) * p(seq_num,:) )';
        temp21 = LR * s(seq_num,:) * ( in(1,:) - w(1,1) * p(seq_num,:) - w(2,1) * s(seq_num,:) )';
        temp22 = LR * s(seq_num,:) * ( in(2,:) - w(1,2) * p(seq_num,:) - w(2,2) * s(seq_num,:) )';
        
        dw(1,:) = dw(1,:) + [temp11 temp12];
        dw(2,:) = dw(2,:) + [temp21 temp22];
        c = z;
        
    end
    
    w(1, :) = w(1, :) + dw(1, :);
    w(2, :) = w(2, :) + dw(2, :);
    
    if(verbose)
        figure(1);
        if(epoch==1)
            disp('epoch 1')
            plot( p(:), s(:), '.', 'MarkerSize', 10, 'MarkerFaceColor', [245,190,65]/255, 'MarkerEdgeColor', [245,190,65]/255 );
            
        else
            hold on            
            plot( p(:), s(:), '.', 'MarkerSize', 10, 'MarkerFaceColor', [0 0 1], 'MarkerEdgeColor', [1 0 0] );
        end
        set(0,'defaultfigurecolor',[1 1 1])        
        set(gca, 'xtick', [0 1.5] )
        set(gca, 'ytick', [-.5 0 1.5] )
        set(gca, 'FontName', 'Calibri')
        
        xlabel('Output 1');
        ylabel('Output 2');
        set(0,'defaultfigurecolor',[1 1 1])
        set(gca, 'FontSize', 14)
        title(['Epoch ' num2str(epoch)]);
    end
    
end
