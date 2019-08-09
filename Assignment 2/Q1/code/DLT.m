function [P_hat] = DLT(image_points, object_points)

	num_points = size(image_points, 2);
	A = zeros(2*num_points, 12);
	for i = 1:num_points
	    A(2*i-1, 5:8)	= 	- object_points(:, i);
	    A(2*i-1, 9:12)	= 	image_points(2, i) * object_points(:, i);
	    A(2*i, 1:4)		= 	- object_points(:, i);
	    A(2*i, 9:12)	= 	image_points(1, i) * object_points(:, i);
	end
	[~, ~, V] = svd(A);
	P_hat = V(:, 12);
	P_hat = reshape(P_hat, [4, 3])';

end
