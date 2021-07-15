function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for ii = 1:size(X,1)
    %比較用の暫定定数を用意する
    %too big numberで初期化する
    temp_length = 10000000000000000;
    temp_idx = 1;
    for jj = 1:K
    %今と暫定のcentroidを比較する
    if temp_length > sum((X(ii,:) - centroids(jj,:)).^2)
    %成立した時は更新する
    temp_idx = jj;
    temp_length = sum((X(ii,:) - centroids(jj,:)).^2);
    end
    end
    idx(ii,1) = temp_idx;
end










% =============================================================

end

