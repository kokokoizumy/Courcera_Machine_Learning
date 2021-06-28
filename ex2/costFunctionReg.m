function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

%仮設関数の定義
hx = sigmoid(X*theta);

%正則化用に、thetaの第一項をゼロに
theta_reg = theta;
theta_reg(1,1) =  0;

%評価関数の定義、第２項に正則化を追加
J = (1/m)*sum(-y.*log(hx)-(1-y).*log(1-hx)) + lambda/(2*m)*sum(theta_reg.*theta_reg);

%偏微分の定義、j=0とそれ以降で式が異なるようにする
grad = (1/m)*(sum((hx - y).*X))' +lambda/m*theta_reg;






% =============================================================

end
