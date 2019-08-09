function plotPose(pose, parts_list)
    connected = zeros(size(parts_list,1), 2, 3);
    for i = 1:size(parts_list,1)
        joint_idx = parts_list(i,1);
        parent_joint_idx = parts_list(i,2);
        joint = pose(joint_idx,:);
        parent_joint = pose(parent_joint_idx,:);
        connected(i,1,:) = parent_joint;
        connected(i,2,:) = joint;
    end
    plot3(connected(:,:,1)', connected(:,:,2)', connected(:,:,3)', 'LineWidth', 4, 'Marker', '*', 'MarkerSize', 6, 'Color', 'red', 'MarkerFaceColor', 'blue', 'MarkerEdgeColor', 'blue')
    axis([-6 6 -6 6 -6 6])
    xlabel('X Axis'); 
    ylabel('Y Axis'); 
    zlabel('Z Axis'); 
end