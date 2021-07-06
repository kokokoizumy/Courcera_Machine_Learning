function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------
%各種値の定義
mval = size(Xval,1)

for ii = 1:m

%訓練データを分割していく
%これで訓練データのii番目までの行を取り出すことができる。
X_train = X((1:ii),:);
y_train = y(1:ii);

%このデータに対して学習をしてもらって、Thetaを受け取るので
Theta_ii = trainLinearReg(X_train,y_train,lambda);

%この値に対して各々学習をしてもらう
%訓練データセットに対する評価関数は
hx_train    = X_train*Theta_ii;
error_train(ii,1) = (1/(2*ii))*sum((hx_train-y_train).^2);

%評価データセットに対する評価関数は
hx_val    = Xval*Theta_ii;  
error_val(ii,1) = (1/(2*mval))*sum((hx_val-yval).^2);
end
% -------------------------------------------------------------

% =========================================================================

end
