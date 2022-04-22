function [inhibwt, c, p , s, vp, vs, vp1, vs1] = recurrent_anti_hebbian_trainer( X, n_epochs, verbose )

% X is a matrix of N by T
% N is the input dim
% T is the number of seqs
% note the elements across the memory and input seq should be correlated
% i.e. element 1 in x and element 1 in c and element 2 in x and element 2
% in c. This is different from Stilp data structure.
% X = [[1.2;2.03;4] [1;2.1;4] [1;2.05;4.1] [1;2.9;4] [1;2.01;4] [1;2;4.6] [1;2;4] [1;2;4] [1;2;4] [1;2;4] ];
% X = (X - mean(X, 2));
num_seqs = size(X,2);
N        = size(X,1);
w = rand(2,2);
% w = [.5 .5; .5 .5];
% w = ones(2,2) - eye(2);
% dw = zeros(2,2);
LR = 0.00001;

c = rand(N, 1);
inhibwt = 0;
% c = zeros(N, 1);

% n_epochs = 1;

start_color = [37,121,133]/255;
end_color  = [255,68,71]/255;
rgb_vec_delta = (end_color - start_color)/(n_epochs*num_seqs);
rgb_vec       = start_color;

start_size = 7;
end_size   = 10;
size_delta = (end_size - start_size)/(n_epochs);
dot_size   = 7;

p = zeros(num_seqs, N);
s = zeros(num_seqs, N);
%     dw = zeros(2,2);

vp = [];
vs = [];
vp1 = [];
vs1 = [];

for epoch = 1 : n_epochs
    
    if(epoch>n_epochs/2), LR = LR/2; end
    
    for seq_num = 1 : num_seqs
        
        x = X(:,seq_num);
        W = x*c';
        z = W*x;
        z = z * (rms(x)/rms(z));
        %         z = min(x, c);
        %         z = 0.9 * x + 0.1 * c;
        in = [c x]';
        %         p(seq_num,:) = w(1,:) * in;
        %         s(seq_num,:) = w(2,:) * in;
        %         temp11 = LR * p(seq_num,:) * ( in(1,:) - w(1,1) * p(seq_num,:) )';
        %         temp12 = LR * p(seq_num,:) * ( in(2,:) - w(1,2) * p(seq_num,:) )';
        %         temp21 = LR * s(seq_num,:) * ( in(1,:) - w(1,1) * p(seq_num,:) - w(2,1) * s(seq_num,:) )';
        %         temp22 = LR * s(seq_num,:) * ( in(2,:) - w(1,2) * p(seq_num,:) - w(2,2) * s(seq_num,:) )';
        
        
        
        p(seq_num,:) = in(1,:) + inhibwt * s(seq_num,:);
        s(seq_num,:) = in(2,:) + inhibwt * p(seq_num,:);
        
        
        denom1 = sqrt(mean(s(seq_num,:).^2));
        denom2 = sqrt(mean(p(seq_num,:).^2));
        
        if(seq_num ==1)
            vp(end+1) = max(p(1,:));
            vs(end+1) = max(s(1,:));
            
            vp1(end+1) = min(p(1,:));
            vs1(end+1) = min(s(1,:));
        end
%         var(p(seq_num,:))
%         var(s(seq_num,:))
        
        dww = -LR*(p(seq_num,:)/denom1)*(s(seq_num,:)/denom2)';
        inhibwt = inhibwt + dww;
        
%                 disp('***');
%                 disp(inhibwt);
        
        
        
        %         temp2 = LR * s(seq_num,:) * ( in - w(1,:)' * p(seq_num,:) - w(2,:)' * s(seq_num,:) );
        
        %         dw(1,:) = dw(1,:) + [temp11 temp12];
        %         dw(2,:) = dw(2,:) + [temp21 temp22];
        c = z;
        
        
        
        %     w(1,:) = w(1,:) + dw(1,:);
        %     w(2,:) = w(2,:) + dw(2,:);
        
        if(verbose && epoch>1)
            figure(1);
            hold on
            %     plot(p(:), s(:), '.', ...
            %         'MarkerSize', 10, 'MarkerEdgeColor', start_color, 'MarkerFaceColor', start_color)
            plot( p(1,:), s(1,:), ...
                '.',...
                'MarkerSize', dot_size, ...
                'MarkerFaceColor', rgb_vec,...
                'MarkerEdgeColor', rgb_vec );
            
            set(0,'defaultfigurecolor',[1 1 1])
            %         set(gca, 'YLim', [-1.5 1.5]);
            %         set(gca, 'XLim', [-1.5 1.5]);
            set(gca, 'xtick', [1.0] )
            set(gca, 'ytick', [-1 1.0] )
            %         xlabel('Component 1');
            %         ylabel('Component 2');
            set(0,'defaultfigurecolor',[1 1 1])
            set(gca, 'FontSize', 14)
            title(['Epoch ' num2str(epoch)]);
        end
        
        %     if(epoch==1 )
        %         input('')
        %     end
        %     pause(.5)
        
        rgb_vec  = rgb_vec + rgb_vec_delta;
        dot_size = dot_size + size_delta;
        
    end
    
end
