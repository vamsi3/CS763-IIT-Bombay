function [ H, params ] = RotationTranslation( p1, p2 )
    p1_prime = p1 - mean(p1, 1);
    p2_prime = p2 - mean(p2, 1);
    
    c = zeros(2, 2);
    n = size(p1, 1);
    for i = 1:n
        c = c + p2_prime(i, :)' * p1_prime(i, :);
    end
    [u, s, v] = svd(c);
    R = u * [1 0; 0 det(u * v')] * v';
    T = mean(p2 - p1 * R', 1);

    H = [[R'; T]'; [0 0 1]];
    theta = atan2(R(2, 1), R(1, 1));
    params = [theta T(1) T(2)];
end
