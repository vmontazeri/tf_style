clear
close all
clc

N = 3e5;
num_clusters = 5;

r = ones(N,1)*0;
g = ones(N,1)*0;
b = ones(N,1)*0;

rgb_ = rand(N,3);
rgb_c = zeros(num_clusters,3);
indx = randsample(N,num_clusters);
for i = 1 : length(indx)
    rgb_(indx(i),:) = rand(1,3);
    rgb_c(i,:) = rgb_(indx(i),:);
end
% rgb_(10,:) = [1,0,0];
% rgb_(200,:) = [0,0,1];
% rgb_(300,:) = [0,1,0];
% rgb_(300,:) = [0,1,0];
% rgb_(300,:) = [0,1,0];
% rgb_(300,:) = [0,1,0];

d = rand(N,2);
% d = [linspace(0,1,1e6); linspace(0,1,1e6)]';

% c1 = d(10,:);
% c2 = d(200,:);
% c3 = d(300,:);

dist_ = zeros(length(indx),1);

for i = 1 : N
%     disp(i)
    if (ismember(i,indx)), continue; end
    for j = 1 : length(indx)
        dist_(j) = min((sqrt(sum( (d(i,:) - d(indx(j),:)).^2 ))^-2), 1e50);
    end
    sum_dist = sum(dist_);
    dist_ = dist_ ./ sum_dist;
    
    rgb_(i,:) = dist_' * rgb_c;
    if sum(isnan(rgb_(i,:))) > 0
        1;
    end
    
end

Bfig(1)
scatter(d(:,1), d(:,2), 20, rgb_, 'filled')
% pause(0.001)
    


