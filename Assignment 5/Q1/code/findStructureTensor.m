function [ H ] = findStructureTensor(I, point, patch_offset, grad_x, grad_y)
    H = zeros(2, 2);
    for x = point(1) - patch_offset : point(1) + patch_offset
        for y = point(2) - patch_offset : point(2) + patch_offset
            grad = double([grad_x(x, y) grad_y(x, y)]);
            H = H + transpose(grad) * grad;
        end
    end
end
