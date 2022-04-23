function [inhibwt, c, p, s] = recurrent_anti_hebbian_trainer( X, c, inhibwt, LR, n_epochs, verbose )
% anti-Hebbian learning
% Required: 
%   X: input, vector of n_features x 1
% Optional
%   n_epochs
%   verbose

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
    p = zeros(num_seqs, num_features);
    s = zeros(num_seqs, num_features);
    
    if(epoch>n_epochs/2), LR = LR/2; end
    
    for seq_num = 1 : num_seqs
        
        x = X(:,seq_num);
        W = x*c';
        z = W*x;
        z = z * (rms(x)/rms(z));
        in = [c x]';       
        
        p(seq_num,:) = in(1,:) + inhibwt * s(seq_num,:);
        s(seq_num,:) = in(2,:) + inhibwt * p(seq_num,:);        
        
        denom1 = sqrt(mean(s(seq_num,:).^2)) + 0;
        denom2 = sqrt(mean(p(seq_num,:).^2)) + 0;
        
        dww = -LR*(p(seq_num,:)/denom1)*(s(seq_num,:)/denom2)';
        inhibwt = inhibwt + dww;
        
        c = z;        
        
        if(verbose)
            figure(1);
            hold on
            
            plot( p(:), s(:), '.', 'MarkerSize', 10, 'MarkerFaceColor', [0 0 1], 'MarkerEdgeColor', [0 0 1] );
            set(0,'defaultfigurecolor',[1 1 1])            
%             set(gca, 'xtick', [1.0] )
%             set(gca, 'ytick', [-1 1.0] )            
            set(0,'defaultfigurecolor',[1 1 1])
            set(gca, 'FontSize', 14)
            title(['Epoch ' num2str(epoch)]);
        end
        
    end
    
end
