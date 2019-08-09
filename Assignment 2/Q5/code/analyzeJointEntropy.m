function [rotation, translation, entropy_grid] = analyzeJointEntropy(moving_image, real_image)

	entropy_grid = zeros(121, 25);

	for i=1:121
		rotated_image = imrotate(moving_image, i - 61, 'crop');
		for j=1:25
			modified_image = imtranslate(rotated_image, [j - 13, 0]);
			modified_image = max(modified_image, 0);
			modified_image = min(modified_image, 255);
			entropy_grid(i, j) = jointEntropy(modified_image, real_image);
		end
	end

	[~, I] = min(entropy_grid(:));
	[rotation, translation] = ind2sub(size(entropy_grid), I);
	rotation = rotation - 61;
	translation = translation - 13;

end
