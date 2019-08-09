clear;
close all;
clc;

%% Your Code Here

INPUT_DIR = "../input/"; % set number of input dir here
N = 247; % set number of input frames here
ROI = [160 257 20 30];
noOfFeatures = 2;
patch_size = 41;
freq = 300;
patch_offset = (patch_size - 1) / 2;

MAX_ITERATIONS = 60;
MIN_DELTA_P_NORM = 0.02;
RMSE_THRESHOLD = 30;
SIGMA = 3;

templates = zeros(noOfFeatures, patch_size, patch_size);
p = zeros(noOfFeatures, 6);

Frames = [];
trackedPoints = [];

[x, y] = ndgrid(0:(patch_size-1), 0:(patch_size-1));
x = x - patch_offset;
y = y - patch_offset;
noOfPoints=numel(x);

for i = 1:N
    I = double(imread(strcat(INPUT_DIR, num2str(i), ".jpg")));
    Frames = [Frames; reshape(I, [1 size(I)])];
    I_smooth = imgaussfilt(I, SIGMA);
    [I_grad_x, I_grad_y] = imgradientxy(I_smooth);

    if rem(i, freq) == 1
        features = selectGoodFeatures(I, noOfFeatures, patch_offset, ROI, I_grad_x, I_grad_y);
        for j = 1:noOfFeatures
            templates(j, :, :) = imcrop(I, [features(j, 1)-patch_offset features(j, 2)-patch_offset patch_size-1 patch_size-1]);
            p(j, :) = [1 0 features(j, 1) 0 1 features(j, 2)];
        end
    else
        for j = 1:noOfFeatures
            template = squeeze(templates(j, :, :));
            count = 1;
            delta_p = ones(1, 6);
            error = 1e5;
            while (count < MAX_ITERATIONS) && (norm(delta_p) > MIN_DELTA_P_NORM) && (error > RMSE_THRESHOLD)
                T = reshape(p(j, :), [3, 2])';
                I_warped = warpImage(I, x, y, T);
                I_grad_x_warped = warpImage(I_grad_x, x, y, T);
                I_grad_y_warped = warpImage(I_grad_y, x, y, T);
                I_error = template - I_warped;

                error = sqrt(sum(sum(I_error.^2)) / (patch_size * patch_size)); % L2 error
                hessian = zeros(6, 6);
                delta_p = zeros(1, 6);
                for k = 1:noOfPoints
                    jacobian = [x(k) y(k) 1 0 0 0; 0 0 0 x(k) y(k) 1]; % Jacobian
                    temp = [I_grad_x_warped(k) I_grad_y_warped(k)] * jacobian;
                    hessian = hessian + temp' * temp; % Hessian
                    delta_p = delta_p + temp * I_error(k);
                end
                delta_p = delta_p / hessian;
                p(j, :) = p(j, :) + delta_p;
                count = count + 1;
            end
            features(j, :) = [p(j, 3) p(j, 6)];
        end
    end
    trackedPoints = [trackedPoints; reshape(features', [1 size(features')])];
end

%% Save all the trajectories frame by frame
% variable trackedPoints assumes that you have an array of size 
% No of frames * 2(x, y) * No Of Features
% noOfFeatures is the number of features you are tracking
% Frames is array of all the frames(assumes grayscale)
noOfPoints = 1;
for i=1:N
    NextFrame = squeeze(Frames(i,:,:));
    imshow(uint8(NextFrame)); hold on;
    for nF = 1:noOfFeatures
        plot(trackedPoints(1:noOfPoints, 1, nF), trackedPoints(1:noOfPoints, 2, nF),'*')
        % [trackedPoints(noOfPoints, 1, nF), trackedPoints(noOfPoints, 2, nF)]
    end
    hold off;
    saveas(gcf,strcat('../output/',num2str(i),'.jpg'));
    close all;
    noOfPoints = noOfPoints + 1;
end 
