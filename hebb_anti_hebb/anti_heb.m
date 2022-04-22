function [W, Y] = anti_heb(n_epochs, n_frames, N2, frame_size, plot_res)

w = rand(2,2);

dw11 = 0; dw12 = 0; dw21 = 0; dw22 = 0;

old_signal = zeros(frame_size, 1);

start_color = [37,121,133]/255;
end_color  = [255,68,71]/255;
rgb_vec_delta = (end_color - start_color)/(n_epochs*n_frames);
rgb_vec       = start_color;

% start_size = 7;
% end_size   = 10;
% size_delta = (end_size - start_size)/(n_epochs*1);
% dot_size   = 7;
epoch_1_color = [244,187,70]/255;
epoch_end_color = [247,110,91]/255;

start_color = [37,121,133]/255;
start_color = [76,150,211]/255;
end_color  = [255,68,71]/255;
end_color = [247,110,91]/255;
% end_color = [175,146,144]/255;
rgb_vec_delta = (end_color - start_color)/(n_epochs);
temp = exp(.03*(1:100));
scale = temp / temp(end);

start_size = 5*3;
end_size   = 5*3;
size_delta = (end_size - start_size)/(n_epochs);
dot_size   = 5*3;

ab_w = [];

new_signal = N2( 1 : 1*frame_size );        
X = [ old_signal new_signal ]';

inhibwt = 0.0;
Y = X;

figure()
hold off
box off
% H=gca;
set(gca,'LineWidth', 3 )

for epoch = 1 : n_epochs
%     LR = 0.001*exp(-epoch/100);
%     LR = 0.001;
%     old_signal = new_signal;

rgb_vec  = start_color * (1-scale(epoch)) + end_color * scale(epoch);
    dot_size = 5*3;
    if(epoch==1)
        rgb_vec = epoch_1_color;
        dot_size = 15*2;
    end
    if(epoch==n_epochs)
        rgb_vec = epoch_end_color;
        dot_size = 10*2;
    end


    for i = 1 : 1

%         new_signal = N2( (i-1)*frame_size+1 : i*frame_size );        

        X = [ N2( (i-1)*frame_size+1 : i*frame_size ) N2( (2-1)*frame_size+1 : 2*frame_size ) ]';
        
%         figure(1)
%         plot(X(1,:), X(2,:), '.')        
        
        Y(1,:) = X(1,:) + inhibwt * Y(2,:);
        Y(2,:) = X(2,:) + inhibwt * Y(1,:);
        
        denom1 = sqrt(mean(Y(2,:).^2));
        denom2 = sqrt(mean(Y(1,:).^2));
        
        if(epoch<100), LR = 0.00001/2; else, LR = 0.0001*2; end
        if epoch==1, LR = 0.00005; end
        dww = -LR*(Y(1,:)/denom1)*(Y(2,:)/denom2)';
        inhibwt = inhibwt + dww;                
        
%         Y = X' * W;
%         dw1 = LR * X(1,:)  * Y;
%         dw2 = LR * X(2,:)  * Y;
%         dw  = [ dw1; dw2 ]/size(X,2);

%         W = (W + dw) ./ repmat( sqrt( sum( (W+dw).^2, 2 ) ), 1, size(W,2));

%         disp('***')
%         disp(dww)
%         disp(inhibwt);

        old_signal = new_signal;
        
        if epoch==1
            temp = Y;
        end

        if(1)            
            figure(1);
            hold on;
%             if(epoch==1&&i==2)
%                 plot( Y(1,:), Y(2,:), '.', 'MarkerSize', end_size, ...
%                     'MarkerFaceColor', rgb_vec,...
%                     'MarkerEdgeColor', rgb_vec );
%             else
                plot( Y(1,:), Y(2,:), '.', 'MarkerSize', dot_size, ...
                    'MarkerFaceColor', rgb_vec,...
                    'MarkerEdgeColor', rgb_vec );
%             end
            %                 axis off;
            set(0,'defaultfigurecolor',[1 1 1])
            set(gca, 'YLim', [-0.45 1.1]);
            set(gca, 'XLim', [-0.4 1.1]);
%             set(gca, 'xtick', [-0.4 0 1 1.1] );
%             set(gca, 'ytick', [-0.4 0 1 1.1] )
            xlabel('Output 1');
        ylabel('Output 2');
        set(0,'defaultfigurecolor',[1 1 1])
        set(gca, 'FontSize', 32)
        box off
        set(gca,'LineWidth', 2 )

            rgb_vec  = rgb_vec + rgb_vec_delta;
            dot_size = dot_size + size_delta;

            pause(0.01);
            if(epoch==1 ) 
%                 input('')
            end
        end    
    end
end


end