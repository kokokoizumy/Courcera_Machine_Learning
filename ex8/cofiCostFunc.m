function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%目的関数の計算
temp_J = ((X*Theta' - Y).^2).*R;
J = 1/2*sum(sum(temp_J));

%勾配の計算をやりたい
%xは映画のジャンルのようなイメージの特徴量
%Thetaはユーザが与えた映画の評価
size_x = size(X)
size_y = size(Y)
size_theta = size(Theta)
size_R = size(R)

%Xの勾配の計算
for ii = 1:size(X,1)
    %R()=1周りの処理
    idx_theta = find(R(ii,:)==1);
    Theta_temp = Theta(idx_theta,:);
    Y_temp = Y(ii,idx_theta);
    
    %勾配の計算
    X_grad(ii,:) = (X(ii,:)*Theta_temp' - Y_temp)*Theta_temp;
end

%Tehtaの勾配の計算
for jj = 1:size(Theta,1)
    %R()=1周りの処理
    idx_x = find(R(:,jj)==1);
    X_temp = X(idx_x,:);
    Y_temp = Y(idx_x,jj);

    size(X_temp*Theta_grad(jj,:)')

    %勾配の計算  1x3
    Theta_grad(jj,:) = (X_temp*Theta(jj,:)' - Y_temp)'*X_temp;
end
% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
