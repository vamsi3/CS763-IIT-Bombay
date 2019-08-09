function [result_pose, composed_rot] = transformPose(rotations, pose, kinematic_chain, root_location)
    % rotations: A 15 x 3 x 3 array for rotation matrices of 15 bones
    % pose: The base pose coordinates 16 x 3.
    % kinematic chain: A 15 x 2 array of joint ordering
    % root_positoin: the index of the root in pose vector.

	composed_rot = eye(4, 4);
	result_pose = transformPoseHelper(rotations, pose, kinematic_chain, root_location, composed_rot);
end

function result_pose = transformPoseHelper(rotations, pose, kinematic_chain, root_location, composed_rot)
	for i = 1:size(kinematic_chain, 1)
		if kinematic_chain(i, 2) == root_location
			transformation_matrix = [[squeeze(rotations(i, :, :)) pose(root_location, :)']; [0 0 0 1]] * [eye(4, 3) [-pose(root_location, :) 1]'];
			pose = transformPoseHelper(rotations, pose, kinematic_chain, kinematic_chain(i, 1), composed_rot * transformation_matrix);
		end
	end

	pose_homogeneous_coordinates = [pose(root_location, :) 1] * composed_rot';
	pose(root_location, :) = pose_homogeneous_coordinates(1:3);
	result_pose = pose;
end
