close all
clc
clear
im = double(imread('../input/chrysler.png'))./255;
figure
imshow(im)

% Test values. Do test for other values too
k1 = 0.1;
k2 = 0.01;
%%
imD = radDist(im, k1, k2);
figure
imshow(imD)

%%
nSteps = 2; % Fill in the number of steps here
imU = radUnDist(imD, k1, k2, nSteps);
figure
imshow(imU)

