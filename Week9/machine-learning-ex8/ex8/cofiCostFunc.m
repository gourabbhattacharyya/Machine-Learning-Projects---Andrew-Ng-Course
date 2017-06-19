function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%--------------------------compute cost function---------------------------
hx = X * Theta';        %hx is 5*4 [X is 5*3 and Theta' is 3*4]
hx_temp = hx;
Y_temp = Y;
for i = 1:num_movies,
    for j = 1:num_users,
        if (R(i,j) == 0),
            hx_temp(i,j) = hx_temp(i,j) * 0;        %remove the value where R = 0
            Y_temp(i,j) = Y_temp(i,j) * 0;          %remove the value where R = 0
        end
    end
end

J = (hx_temp - Y_temp) .^ 2;         %J is 5*4
J = sum(sum(J))/2;              %J is scalar value

%--------------------------Completed---------------------------------------

%--------------------------compute gradient--------------------------------

X_grad = X_grad + ((hx_temp - Y_temp) * Theta);    %X_grad is 5*3 [(hx_temp - Y_temp) is 5*4 and theta is 4*3]

Theta_grad = Theta_grad + ((hx_temp - Y_temp)' * X);   %Theta_grad is 4*3 [(hx_temp - Y_temp)' is 4*5 and X is 5*3]

%--------------------------Completed---------------------------------------

%--------------------------compute regularized cost function---------------------------

J = J + (lambda * (sum(sum(Theta .^ 2)) + sum(sum(X .^ 2)))) / 2;

%--------------------------Completed---------------------------------------

%--------------------------compute regularized gradient---------------------------

X_grad = X_grad + lambda .* X;          %X_grad is 5*3 [X_grad is 5*3 and lambda*X is 5*3]
Theta_grad = Theta_grad + lambda .* Theta;  %Theta_grad is 4*3 [Theta_grad is 4*3 and lambda*Theta is 4*3]

%--------------------------Completed---------------------------------------

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
