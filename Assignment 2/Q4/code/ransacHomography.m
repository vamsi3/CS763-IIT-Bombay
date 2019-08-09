function [ H ] = ransacHomography( x1, x2, thresh )
	
	n = size(x1, 1);
	x1_homogeneous = [x1 ones(n, 1)]';
	x2_homogeneous = [x2 ones(n, 1)]';
	best_inlier_count = 0;
	for i = 1:240
		sample_indices = randperm(n, 4);
		H_prime = homography(x1(sample_indices, :), x2(sample_indices, :));
		p = H_prime * x1_homogeneous;
		p = p ./ p(3, :);
		inlier_count = sum(sum((x2_homogeneous - p).^2) < thresh);
		if inlier_count > best_inlier_count
			best_inlier_count = inlier_count;
			H = H_prime;
		end
	end

end
