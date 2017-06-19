function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%---------------------calculate regularized LR CostFunction----------------
hx = X * theta;
J = J + ((hx - y) .^ 2);
J = sum(J/(2 * m));

regularFactor = theta(2:length(theta),:)' .^ 2;
regularFactor = sum(regularFactor);
regularFactor = (lambda * regularFactor)/(2 * m);

J = J + regularFactor;

%-------------------------------------completed----------------------------

%---------------------calculate regularized LR gradient--------------------

temp_grad = (X' * (hx - y));   %temp_grad is 2*1 [X' is 2*12 and (hx - y) is 12*1]

grad(1,:) = (grad(1,:) + sum(temp_grad(1,:)))/m;

for i = 2:length(grad),
    grad(i,:) = grad(i,:) + sum(temp_grad(i,:));
    grad(i,:) = (grad(i,:) + (lambda * theta(i,:)))/m;
end

%-------------------------------------completed----------------------------



% =========================================================================

grad = grad(:);

end
