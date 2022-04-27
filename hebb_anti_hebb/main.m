clear
close all
clc

c = randn(640, 1);
w = randn(500,640);
inhibwt = 0;
LR = 1e-4;

path_ = 'E:\databases\IEEE\wideband\';
addpath('backpropagation_algorithm-master')
addpath('C:\Users\mntiv\Desktop\projects\neural-style-tf\Sound_Texture_Synthesis_Toolbox_v1.7\Sound_Texture_Synthesis_Toolbox\');
addpath(path_);


net = create_nn([640,250,640], @sigmoid_function);

wav_files = dir(path_);
wav_files = {wav_files.name};

p_ = zeros(500, 640);
counter = 1;
for i = 1 : 100
for wav_ = wav_files
    if strcmp(char(wav_), '.') || strcmp(char(wav_), '..') || strcmp(char(wav_), '.DS_Store'), continue; end
    if ~contains(char(wav_), '.wav'), continue; end
%     disp(wav_);
    mod_ = get_mods([path_ char(wav_)], 0);
    mod_vec = mod_(:);
    mod_vec = mod_vec + 0.05*abs(randn(size(mod_vec)));
    
%     mod_vec = mod_vec - mean(mod_vec);
%     mod_vec = mod_vec / (std(mod_vec) + eps);
%     mod_vec = mod_vec / max(abs(mod_vec));
    
    [net,~,~,loss] = train(net, mod_vec', mod_vec', @cost_function, 0.9);
    Bfig(1)
    temp = abs([net(1).W net(2).W']);
    imagesc(temp)
%     mean(temp(:))
%     mean(abs(loss))
    [a,~] = max(temp);
    [c,d] = max(a);
    [e,f] = max(temp(:,d));
    disp([num2str(f) ', ' num2str(d) ' mean coeffs ' num2str(mean(temp(:)))])
%     hold on
%     scatter(f,d, randsample(30:5:50,1), [1 0 0], 'filled');
%     hold off
    pause(0.5)
    
    
%     X = mod_vec;
%     for i = 1 : 1
% %         X = mod_vec + rand(size(mod_vec));
%         [w, c, p , s] = recurrent_pca_trainer( X, c, w, LR, 100, 1 );        
%         %      [inhibwt, c, p, s] = recurrent_anti_hebbian_trainer( X, c, inhibwt, LR, 1000, 1 );
%     end
%     p_(counter, : ) = p;    
    counter = counter + 1;
end
end
