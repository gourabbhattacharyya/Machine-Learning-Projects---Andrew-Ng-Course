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

for i = 1:m,
    z = theta' * X(i,:)';
    hx = sigmoid(z);
    J_temp = -1 * ((y(i,:) * log(hx)) + ((1 - y(i,:))' * log(1 - hx)'));
    J = J + J_temp;
    
    grad = grad + (hx - y(i,:))' * X(i,:)';
end
J = J/m;
grad = (grad)/m;

regTheta = 0;
for j = 2:size(X,2),
    regTheta = regTheta + (theta(j,:) ^ 2);
end
regTheta = (lambda * regTheta)/(2 * m);
J = J + regTheta;

for k = 2:size(grad,1),
    grad(k,:) = grad(k,:) + ((lambda * theta(k,:))/m);


% =============================================================

end
