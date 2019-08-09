function [ params ] = RANSAC( x1, x2, thresh, transformType )
    
    n = size(x1, 1);
    x1_homogeneous = [x1 ones(n, 1)]';
    x2_homogeneous = [x2 ones(n, 1)]';
    best_inlier_count = 0;
    num_tries = 250;
    num_samples = 4;
    for i = 1:num_tries
        sample_indices = randperm(n, num_samples);
        if transformType == "Rotation_Translation"
            [H_prime, params_prime] = RotationTranslation(x1(sample_indices, :), x2(sample_indices, :));
        end
        p = H_prime * x1_homogeneous;
        p = p ./ p(3, :);
        inlier_count = sum(sum((x2_homogeneous - p).^2) < thresh);
        if inlier_count > best_inlier_count
            best_inlier_count = inlier_count;
            H = H_prime;
            params = params_prime;
        end
    end

end
