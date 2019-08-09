function [ H ] = homography( p1, p2 )

	number_of_points = size(p1, 1);
	A = zeros(2*number_of_points, 9);
	for i = 1:number_of_points
		A(2*i-1, :) = [-p1(i, :) -1 0 0 0 p2(i, 1)*p1(i, :) p2(i, 1)];
		A(2*i, :)   = [0 0 0 -p1(i, :) -1 p2(i, 2)*p1(i, :) p2(i, 2)];
	end
	[~, ~, V] = svd(A);
	P = V(:, 9);
	H = reshape(P, [3, 3])';

end
