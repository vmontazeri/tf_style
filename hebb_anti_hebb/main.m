clear
close all
clc

c = randn(640, 1);
w = randn(2,2);
inhibwt = 0;
LR = 1e-4;

path_ = 'C:\Users\mntiv\Desktop\projects\neural-style-tf\Sound_Texture_Synthesis_Toolbox_v1.7\Sound_Texture_Synthesis_Toolbox\Example_Textures\';
addpath('C:\Users\mntiv\Desktop\projects\neural-style-tf\Sound_Texture_Synthesis_Toolbox_v1.7\Sound_Texture_Synthesis_Toolbox\');
addpath(path_);

wav_files = dir(path_);
wav_files = {wav_files.name};

p_ = zeros(500, 640);
counter = 1;
for wav_ = wav_files
    if strcmp(char(wav_), '.') || strcmp(char(wav_), '..') || strcmp(char(wav_), '.DS_Store'), continue; end
    disp(wav_);
    mod_ = get_mods([path_ char(wav_)], 1);
    mod_vec = mod_(:);
    
    mod_vec = mod_vec - mean(mod_vec);
    mod_vec = mod_vec / (std(mod_vec) + eps);
    
    X = mod_vec;
    for i = 1 : 1
%         X = mod_vec + rand(size(mod_vec));
        [w, c, p , s] = recurrent_pca_trainer( X, c, w, LR, 100, 1 );        
        %      [inhibwt, c, p, s] = recurrent_anti_hebbian_trainer( X, c, inhibwt, LR, 1000, 1 );
    end
    p_(counter, : ) = p;    
    counter = counter + 1;
end