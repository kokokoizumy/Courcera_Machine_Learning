function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%仮設関数の計算
hx = X*theta;

%ロジスティック回帰用にsigmoid関数をかます
hx =sigmoid(hx)

%評価関数の計算を行う
J = (1/m)*sum(-y.*log(hx)-(1-y).*log(1-hx));

%評価関数の偏微分の計算
grad = (1/m)*(sum((hx - y).*X))'









% =============================================================

end
