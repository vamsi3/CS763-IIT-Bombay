function rot_matrix = angles2rot(angles_list)
	%% Your code here
	% angles_list: [theta1, theta2, theta3] about the x,y and z axes,
	% respectively.

	n_parts = size(angles_list, 1);
	rot_matrix = zeros(n_parts, 3, 3);
	for i = 1:n_parts
		theta1 = angles_list(i, 1); theta2 = angles_list(i, 2); theta3 = angles_list(i, 3);
		Rx = [1 0 0; 0 cosd(theta1) -sind(theta1); 0 sind(theta1) cosd(theta1)];
		Ry = [cosd(theta2) 0 sind(theta2); 0 1 0; -sind(theta2) 0 cosd(theta2)];
		Rz = [cosd(theta3) -sind(theta3) 0; sind(theta3) cosd(theta3) 0; 0 0 1];
		rot_matrix(i, :, :) = Rz * Ry * Rx;
	end
end
