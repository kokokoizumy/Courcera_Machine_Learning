function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

vec_fp = zeros(size(yval));
vec_fp = zeros(size(yval));
vec_fn = zeros(size(yval));


stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

%正誤判定を行う、小さいのであれば誤(=1)大きいのであれば正(=0)
prediction  = (pval<epsilon);

for ii = 1:size(pval,1)
    vec_tp(ii) = (prediction(ii) == 1)&&(yval(ii) == 1);
    vec_fp(ii) = (prediction(ii) == 0)&&(yval(ii) == 1);
    vec_fn(ii) = (prediction(ii) == 1)&&(yval(ii) == 0);
end

tp = nnz(vec_tp);
fp = nnz(vec_fp);
fn = nnz(vec_fn);

prec = tp/(tp+fp);
rec  = tp/(tp+fn);

F1 = 2*(prec*rec)/(prec+rec);


    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
