tic;

% Set image data directory here.
data_dir = "../input/ledge";

% A lot of syntax used here is inspired directly from:
% https://in.mathworks.com/help/vision/examples/feature-based-panoramic-image-stitching.html

image_data = imageDatastore(data_dir);

num_images = numel(image_data.Files);
image_sizes = zeros(num_images, 2);
transforms(num_images) = projective2d(eye(3));

I = rgb2gray(readimage(image_data, 1));
image_sizes(1, :) = size(I);
points = detectSURFFeatures(I);
[features, points] = extractFeatures(I, points);

for i = 2:num_images
	points_prev = points;
	features_prev = features;
	
	I = rgb2gray(readimage(image_data, i));
	image_sizes(i, :) = size(I);
	points = detectSURFFeatures(I);
	[features, points] = extractFeatures(I, points);

	index_pairs = matchFeatures(features, features_prev, 'Unique', true);

	matched_points = points(index_pairs(:, 1), :).Location;
	matched_points_prev = points_prev(index_pairs(:, 2), :).Location;

	transforms(i) = projective2d(ransacHomography(matched_points, matched_points_prev, 2)');
	transforms(i).T = transforms(i).T * transforms(i-1).T;
end


% Empty Panorama initialization

for i = 1:num_images
	[xlim(i, :), ylim(i, :)] = outputLimits(transforms(i), [1 image_sizes(i, 2)], [1 image_sizes(i, 1)]);
end

xlim_mean = mean(xlim, 2);
[~, idx] = sort(xlim_mean);
center_idx = floor((numel(transforms)+1)/2);
center_image_idx = idx(center_idx);
center_image_inverse_transform = invert(transforms(center_image_idx));

for i = 1:num_images
	transforms(i).T = transforms(i).T * center_image_inverse_transform.T;
end

for i = 1:num_images
	[xlim(i, :), ylim(i, :)] = outputLimits(transforms(i), [1 image_sizes(i, 2)], [1 image_sizes(i, 1)]);
end

black_border = [33, 33];
image_size_max = max(image_sizes);
x_min = min([1; xlim(:)]) - black_border(1);
x_max = max([image_size_max(2); xlim(:)]) + black_border(1);
y_min = min([1; ylim(:)]) - black_border(2);
y_max = max([image_size_max(1); ylim(:)]) + black_border(2);

width  = round(x_max - x_min);
height = round(y_max - y_min);
panorama = zeros([height width 3], 'like', I);


% Panorama creation

blender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');
panorama_view = imref2d([height width], [x_min x_max], [y_min y_max]);

for i = 1:num_images
    I = readimage(image_data, i);
    warped_image = imwarp(I, transforms(i), 'OutputView', panorama_view);
    mask = imwarp(true(size(I, 1), size(I, 2)), transforms(i), 'OutputView', panorama_view);
    panorama = step(blender, panorama, warped_image, mask);
end

figure;
imshow(panorama);

toc;
