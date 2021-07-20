function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

%各々の特徴量の平均を計算する
%size(mu)
temp_mu = 1/m*sum(X);
mu = temp_mu' 
%size(mu)

%分散の計算、for-loopで実装したけどもう少しいい方法はあるかもな。
for ii = 1:n
    temp = 0;
    
    for jj = 1:m
        temp = temp + (X(jj,ii) - mu(ii,1))^2;
    end

    temp2 = 1/m*temp;
    sigma2(ii,1) = temp2;

end

% =============================================================


end
