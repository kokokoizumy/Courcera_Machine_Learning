function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%svmPredictを使うための準備
x1 = [1 2 1]; x2 = [0 4 -1];


%探索範囲の準備
C_Vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_Vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
num_c = length(C_Vec);
num_s = length(sigma_Vec);

result = zeros(num_c,num_s);

%for loop二つで探索を行う
for ii = 1:num_c
C = C_Vec(ii);

for jj = 1:num_s
sigma = sigma_Vec(jj);

%modelの定義これでいいのか悪いのか？
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 

%関数simPredictで値の推測をしてもらう
predictions = svmPredict(model,Xval);

%結果を得る
result(ii,jj) =  mean(double(predictions ~= yval));
end
end

[val,id]=min(result(:));
[ii,jj] = ind2sub(size(result),id);

C = C_Vec(ii);
sigma = sigma_Vec(jj);

% =========================================================================

end
