function [ points ] = selectGoodFeatures(I, num_points, patch_offset, ROI, I_grad_x, I_grad_y)
    points = detectHarrisFeatures(I);
    points = double(round(points.Location));
    total_points = size(points, 1);
    min_eigenvalues = ones(total_points, 2) * -1;

    for i=1:total_points
        if (points(i, 1) >= ROI(1) && points(i, 1) <= ROI(1) + ROI(3) && points(i, 2) >= ROI(2) && points(i, 2) <= ROI(2) + ROI(4))
            eigenvalues = eig(findStructureTensor(I, points(i, :), patch_offset, I_grad_x, I_grad_y));
            min_eigenvalues(i) = min(eigenvalues);
        end
    end
    [~, sorted_indices] = sort(min_eigenvalues, 'descend');
    points = points(sorted_indices, :);
    points = points(1:num_points, :);
end
