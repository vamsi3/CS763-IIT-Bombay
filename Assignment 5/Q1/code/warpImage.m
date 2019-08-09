function [ I_warped ] = warpImage(I, x, y, T)    
    x_prime = round(T(1, 1) * x + T(1, 2) * y + T(1, 3));
    y_prime = round(T(2, 1) * x + T(2, 2) * y + T(2, 3));
    linear_index = 1 + y_prime + x_prime * size(I, 1);
    I_warped = I(linear_index);
end
