tic;

% Finding minimum entropy

[x_plot, y_plot] = meshgrid(-12:12, -60:60);

barbara = imread('../input/barbara.png');
negative_barbara = imread('../input/negative_barbara.png');
modified_negative_barbara = modifyMovingImage(negative_barbara);
[barbara_rotation, barbara_translation, barbara_entropy_grid] = analyzeJointEntropy(modified_negative_barbara, barbara);
figure; surf(x_plot, y_plot, barbara_entropy_grid);

flash = rgb2gray(imread('../input/flash1.jpg'));
no_flash = rgb2gray(imread('../input/noflash1.jpg'));
modified_no_flash = modifyMovingImage(no_flash);
[flash_rotation, flash_translation, flash_entropy_grid] = analyzeJointEntropy(modified_no_flash, flash);
figure; surf(x_plot, y_plot, flash_entropy_grid);

% Misaligned minimum entropy case

misaligned_steps = 4; % must be divisor of 40 for proper code working.
[misaligned_barbara_rotation, misaligned_barbara_translation, misaligned_barbara_entropy_grid] = findMisalignedMinimum(modified_negative_barbara, barbara, misaligned_steps);
[misaligned_x_plot, misaligned_y_plot] = meshgrid(-180:misaligned_steps:180, -260:misaligned_steps:260);
figure; surf(misaligned_x_plot, misaligned_y_plot, misaligned_barbara_entropy_grid);

% Image alignment

corrected_negative_barbara = imrotate(imtranslate(modified_negative_barbara, [barbara_translation, 0]), barbara_rotation, 'crop');
corrected_no_flash = imrotate(imtranslate(modified_no_flash, [flash_translation, 0]), flash_rotation, 'crop');

figure;	imshow(imfuse(negative_barbara, corrected_negative_barbara, 'blend','Scaling','none'));
figure;	imshow(imfuse(no_flash, corrected_no_flash, 'blend','Scaling','none'));

figure;	imshow(imfuse(negative_barbara, corrected_negative_barbara, 'diff','Scaling','none'));
figure;	imshow(imfuse(no_flash, corrected_no_flash, 'diff','Scaling','none'));

toc;
