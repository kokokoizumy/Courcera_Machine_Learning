function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%①まずはForwardPropagationで、各層のアクティベーションを計算する
%一層目
X = [ones(m,1) X];
z_1 = X*Theta1';
a_2 = sigmoid(z_1);

%二層目：出力層
a_2 = [ones(m,1) a_2];
z_2 =  a_2*Theta2';
a_3 = sigmoid(z_2);

%目標関数用に設定しておく。
hx  = a_3;

%正則化用に、Theta1,Theta2の先頭をゼロにする
%バイアス項を変更するように、
Theta1_reg = Theta1;
Theta1_reg(:,1) =  0;

Theta2_reg = Theta2;
Theta2_reg(:,1) =  0;


%評価関数の計算
%データセットに対するループは、ベクトル化で行う
%10個あるラベルに対する計算は、for loopで行う。

for ii = 1:num_labels

%出力関数を再ラベリングする
yy = (y == ii);

%出力層の各々に対して目標関数を追加する
J = J + (1/m)*sum(-yy.*log(hx(:,ii))-(1-yy).*log(1-hx(:,ii))); 

%誤差逆伝播法用に出力層の値を用意する
if ii ==1
    y_k = yy;
else
    y_k = [y_k yy];
end

end

%正則化の項を追加する
J = J + lambda/(2*m)*(sum(sum(Theta1_reg.^2)) + sum(sum(Theta2_reg.^2)));

% -------------------------------------------------------------

%誤差逆伝播法を実装する
%初めてなので、ちょっと泥臭くやるか。

%⓪誤差項DELTAの初期化
DELTA2 = zeros(size(Theta2));
DELTA1 = zeros(size(Theta1));
delta3 = zeros(num_labels,1);
%delta2 = zeros(26,1);

for ii = 1:m
%①出力層の計算はしているので注目する値を取り出すのみ
%行のベクトルなので、列ベクトルに修正す

%入力層
a1 = X(ii,:)';

%第二層への入力
z1 = z_1(ii,:)';

%第二層のアクティベーション関数
a2 = a_2(ii,:)';

%出力層
a3 = a_3(ii,:)';

    for iii = 1:num_labels
        %誤差を計算する
        delta3(iii,1) = a3(iii) - (y(ii,1)==iii);
    end

%③隠れ層２層目を計算する
%delta3は10*1の列ベクトル
%Theta2は10*26の行列
%sigmoidGradient(a_2)は10*1の行列
%size(Theta2)
%size(delta3)
delta2 = (Theta2'*delta3).* (a2.*(1-a2));

%④隠れ層1層目はないので、次は普通にdeltaの計算をしていく。
%DELTA2は10*26の行列
%delta3は10*1の列ベクトル、a2は26*1の列ベクトル
%DELTA1は25*401の行列
%delta2(2:end)は25*1の列ベクトル
%a1は401*1の列ベクトル
DELTA2 = DELTA2 + delta3 * a2';
DELTA1 = DELTA1 + delta2(2:end) * a1';

%⑤Thetaの微分値を計算する
Theta2_grad = (1/m)*DELTA2;
Theta1_grad = (1/m)*DELTA1;

end
% =========================================================================

% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
