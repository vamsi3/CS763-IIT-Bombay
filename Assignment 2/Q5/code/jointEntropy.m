function [entropy_value] = jointEntropy(moving_image, real_image)

	overlap_indices = moving_image ~= 0 & real_image ~= 0;
	moving_image = moving_image(overlap_indices);
	real_image = real_image(overlap_indices);
	moving_image = floor(moving_image(:) / 10) + 1;
	real_image = floor(real_image(:) / 10) + 1;

	len = length(moving_image);
	histogram = zeros(26, 26);

	for i=1:len
		histogram(moving_image(i), real_image(i)) = 1 + histogram(moving_image(i), real_image(i)); 
	end

	histogram = histogram / len;
	histogram = histogram(histogram > 0); % ensure no zeros go into log
	entropy_value = histogram .* log(histogram); 
	entropy_value = -sum(entropy_value(:));
	
end
