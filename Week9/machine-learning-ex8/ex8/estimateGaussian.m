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

for j = 1:n,
    temp_mu = 0;
    temp_mu = temp_mu + sum(X(1:m,j));   %temp_mu is scalar value[X is 307*2 matrix]
    temp_mu = temp_mu/m;
    
    mu(j,:) = temp_mu;
    
    temp_sig = 0;
    temp_sig = temp_sig + sum((X(1:m,j) - temp_mu) .^ 2);   %temp_sig is scaler value[X is 307*2 matrix]
    temp_sig = temp_sig/m;
    
    sigma2(j,:) = temp_sig;
end
% =============================================================


end
