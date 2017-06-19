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

CList = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmaList = [0.01 0.03 0.1 0.3 1 3 10 30];

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

cLength = length(CList);
sigmaLength = length(sigmaList);
error_summary = [];
%--------------------------calculate model,predictions and error from X, y,
%Xval, yval----------------------------------------------------
for i = 1:cLength,
    for j = 1:sigmaLength,
        model= svmTrain(X, y, CList(:,i), @(x1, x2) gaussianKernel(x1, x2, sigmaList(:,j)));
        predictions = svmPredict(model, Xval);
        perror = mean(double(predictions ~= yval));
        error_summary = [error_summary ; CList(:,i), sigmaList(:,j), perror];       %store the output in the matrix
    end
end
disp(error_summary);
[minError, indexVal] = min(error_summary(:,3));         %get the index of min error from column 3

C = error_summary(indexVal,1);
sigma = error_summary(indexVal,2);

%---------------------------------Completed--------------------------------
% =========================================================================

end
