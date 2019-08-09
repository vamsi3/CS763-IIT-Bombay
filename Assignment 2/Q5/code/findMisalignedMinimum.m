function [rotation, translation, entropy_grid] = findMisalignedMinimum(moving_image, real_image, steps)

	entropy_grid = zeros(1 + 520/steps, 1 + 360/steps);

	for i=1:steps:521
		rotated_image = imrotate(moving_image, i - 261, 'crop');
		for j=1:steps:361
			modified_image = imtranslate(rotated_image, [j - 181, 0]);
			modified_image = max(modified_image, 0);
			modified_image = min(modified_image, 255);
			entropy_grid(1 + (i-1)/steps, 1 + (j-1)/steps) = jointEntropy(modified_image, real_image);
		end
	end

	[~, I] = min(entropy_grid(:));
	[rotation, translation] = ind2sub(size(entropy_grid), I);
	rotation = rotation - 261;
	translation = translation - 181;

end
