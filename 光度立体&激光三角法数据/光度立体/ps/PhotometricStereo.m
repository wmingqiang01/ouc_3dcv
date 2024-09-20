function [rho, n] = PhotometricStereo(I, mask, L)

% [rho, n] = PhotometricStereo(I, mask, L)
%
% INPUT:
%   I: N1xN2xM array, with each level I(:,:,i) the i-th intensity image.
%   mask: N1xN2xM boolean array, with each level mask(:,:,i) the shadow mask
%         of image I(:,:,i), 0 for pixel being in shadow and 1 otherwise.
%   L: 3xM array for calibrated lighting.
%
% OUTPUT:
%   rho: N1xN2 matrix for the estimated albedo map.
%   n: N1xN2x3 array for the estimated normal map.

% Resize the input to MxN.
[N1, N2, M] = size(I);
N = N1 * N2;
I = reshape(I, [N, M])';  % Transpose to MxN for matrix operations
mask = reshape(mask, [N, M]);

% Create a mask index for efficient computation.
maskIndex = zeros(N, 1);
for i = 1:M
  maskIndex = maskIndex * 2 + mask(:,i);
end
uniqueMaskIndices = unique(maskIndex);

% Initialize b for storing scaled normals.
b = nan(3, N);
for iIdx = 1:length(uniqueMaskIndices)
  idx = uniqueMaskIndices(iIdx);
  % Find all pixels with this index.
  pixelIdx = find(maskIndex == idx);
  % Find all images that are active by this index.
  imageTag = mask(pixelIdx(1), :);
  if (sum(imageTag) < 2)
    continue;
  end
  % Create a 3xM' lighting matrix L.
  Li = L(:, imageTag);
  % Create an M'xN' matrix of image intensities.
  Ii = I(imageTag, pixelIdx);
  % Compute the scaled normal using least squares.
  b(:, pixelIdx) = Li' \ Ii;
end

% Reshape b back to its original dimensions.
b = reshape(b', [N1, N2, 3]);

% Compute albedo (rho) and normal vectors (n).
rho = sqrt(sum(b.^2, 3));  % Magnitude of the vector gives the albedo
n = b ./ rho;              % Normalize b to get the unit normal

% Handle cases where rho is zero to avoid division by zero.
rho(isnan(rho)) = 0;
n(isnan(n)) = 0;

end
