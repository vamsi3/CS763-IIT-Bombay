% Q2data contains
% corners 			: 3x2 double - Contains three 2D pixel coordinates of three visible playable area of the stadium.
% p1		 		: 4x2 double - Contains four 2D pixel coordinates of four corners of dee.
% p2			 	: 4x2 double - Contains four 2D coordinates of four corners of dee in our assumed transformed system (defined as top-view).
load('../input/Q2data');

H = homography(p1, p2);
transformed_corners = H * [corners'; [1 1 1]];
for i = 1:3
	transformed_corners(:, i) = transformed_corners(:, i) / transformed_corners(3, i);
end
length = norm(transformed_corners(:, 1) - transformed_corners(:, 2))
width  = norm(transformed_corners(:, 2) - transformed_corners(:, 3))
