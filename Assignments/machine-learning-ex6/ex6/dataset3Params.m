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

error_pred = zeros(64, 1);
C = [.01; .03; .1; .3; 1; 3; 10; 30];  
sigma = [.01; .03; .1; .3; 1; 3; 10; 30].^.5;  

for i = 1:8
 for j = 1:8
model= svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j)));
% model= svmTrain(Xval, yval, C(i), @(x1, x2) gaussianKernel(Xval(:, 1), Xval(:, 2), sigma(j))); 
predictions = svmPredict(model, Xval);
error_pred(j+(8*i-8)) = mean(double(predictions ~= yval));
  endfor
endfor

[op_val, op_ind] = min(error_pred);
C = C(idivide(op_ind, 8)+1);
sigma = sigma(rem(op_ind, 8));

% =========================================================================

end
