clear
close all
clc

file_name = 'mel_features_img_relu5_1.mat';
load(file_name)

feature_space = squeeze(feature_space);

% Bfig(1)
% imagesc(feature_space)
size(feature_space)
Bfig(1)
for i = 1 : size(feature_space,3)
imagesc(feature_space(:,:,i))
title(i)
input('')
end

%%
clear
close all
clc

load('test.mat')
feature_space = mel;

% mel_ = squeeze(mel(1,:,:))';
% mel_ = [mel_ squeeze(mel(2,:,:))' squeeze(mel(3,:,:))' squeeze(mel(4,:,:))'];
% 
% Bfig(1)
% imagesc( mel_ )
% set(gca, 'YDir', 'normal')

size(feature_space)
Bfig(1)
for i = 1 : size(feature_space,3)
imagesc(feature_space(:,:,i)')
title(i)
input('')
end

