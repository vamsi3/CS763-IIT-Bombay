%% Read Video & Setup Environment
clear
clc
close all hidden
[FileName,PathName] = uigetfile({'*.avi'; '*.mp4'},'Select shaky video file');

cd mmread
vid=mmread(strcat(PathName,FileName));
cd ..
s=vid.frames;

%% Your code here

outV = s;
[h w] = size(rgb2gray(s(1).cdata));
N = size(s, 2) - 1;
k = 10; % Half Window size
motion_estimate = zeros(N + 1, 3);

for i = 1:N
    I_curr = rgb2gray(s(i).cdata);
    I_next = rgb2gray(s(i + 1).cdata);
    points_curr = detectSURFFeatures(I_curr);
    points_next = detectSURFFeatures(I_next);
    [features_curr, valid_points_curr] = extractFeatures(I_curr, points_curr);
    [features_next, valid_points_next] = extractFeatures(I_next, points_next);
    index_pairs = matchFeatures(features_curr, features_next);
    matched_points_curr = valid_points_curr(index_pairs(:, 1), :);
    matched_points_next = valid_points_next(index_pairs(:, 2), :);
    motion_estimate(i + 1, :) = motion_estimate(i, :) + RANSAC(matched_points_next.Location, matched_points_curr.Location, 1.5, "Rotation_Translation");
end

smooth_motion_estimate = zeros(N + 1, 3);

for i=1:N+1
    smooth_motion_estimate(i, :) = mean(motion_estimate(max(1, i - k):min(N, i + k), :), 1);
    p = motion_estimate(i, :) - smooth_motion_estimate(i, :);
    outV(i).cdata = imwarp(s(i).cdata, affine2d([cos(p(1)) sin(p(1)) 0; -sin(p(1)) cos(p(1)) 0; p(2) p(3) 1]), 'OutputView', imref2d(size(s(i).cdata)));
end

%% Plots
[~, name, ~] = fileparts(FileName);

figure;
hold on;
plot(1:N+1, motion_estimate(:, 1),'color','blue');
plot(1:N+1, smooth_motion_estimate(:, 1),'color','red');
title('Cumulative Rotation');
legend('Original','Smooth');
hold off;
saveas(gcf, strcat(PathName, 'plot_theta_', name, '.jpg'));

figure;
hold on;
plot(1:N+1, motion_estimate(:, 2),'color','blue');
plot(1:N+1, smooth_motion_estimate(:, 2),'color','red');
title('Cumulative Translation X');
legend('Original','Smooth');
hold off;
saveas(gcf, strcat(PathName, 'plot_tx_', name, '.jpg'));

figure;
hold on;
plot(1:N+1, motion_estimate(:, 3),'color','blue');
plot(1:N+1, smooth_motion_estimate(:, 3),'color','red');
title('Cumulative Translation Y');
legend('Original','Smooth');
hold off;
saveas(gcf, strcat(PathName, 'plot_ty_', name, '.jpg'));

%% Write Video
vfile=strcat(PathName,'combined_',FileName);
ff = VideoWriter(vfile);
ff.FrameRate = 30;
open(ff)

for i=1:N+1
    f1 = s(i).cdata;
    f2 = outV(i).cdata;
    vframe=cat(1,f1, f2);
    writeVideo(ff, vframe);
end
close(ff)

%% Display Video
figure
msgbox(strcat('Combined Video Written In ', vfile), 'Completed') 
displayvideo(outV,0.01)
