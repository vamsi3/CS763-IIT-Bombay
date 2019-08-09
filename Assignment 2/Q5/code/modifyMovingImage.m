function [modified_moving_image] = modifyMovingImage(moving_image)

	modified_moving_image = imrotate(moving_image, 23.5, 'crop');
	modified_moving_image = imtranslate(modified_moving_image, [-3 0]);
	modified_moving_image = modified_moving_image + uint8(randi(9, size(modified_moving_image))) - 1;
	modified_moving_image = max(modified_moving_image, 0);
	modified_moving_image = min(modified_moving_image, 255);

end
