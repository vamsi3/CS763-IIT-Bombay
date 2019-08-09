function [normalized_points, T] = normalizePoints(points)

	[dim_points, num_points] = size(points);
	centroid = mean(points, 2);
	scale_factor = sqrt(dim_points) / mean(sqrt(sum((points - centroid) .^ 2)));
	T = scale_factor * [eye(dim_points), -centroid];
	T = [T; [zeros(1, dim_points) 1]];
	normalized_points = [points; ones(1, num_points)];
	normalized_points = T * normalized_points;

end
