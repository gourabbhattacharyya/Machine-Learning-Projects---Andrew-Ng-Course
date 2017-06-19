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

%------------------------Unregularized Cost Function-----------------------
X = [ones(m, 1) X];     %X is 5000*401 dimension
z_layer2 = X * Theta1';     %layer2 is 5000*25 dimension(Theta1' is 401*25)
a_layer2 = sigmoid(z_layer2);      %a2 has 5000*25 dimension
a_layer2 = [ones(m, 1) a_layer2];   %a2 has 5000*26 dimension

z_layer3 = a_layer2 * Theta2';  %layer3 is 5000*10 dimension(a_layer2 is 5000*26 and Theta2' is 26*10)
a_layer3 = sigmoid(z_layer3);    %a3 has 5000*10 dimension

%disp(a_layer3);

for i = 1:m,
    hx = a_layer3(i,:);
    sum = 0;
    
    %make logical array of y
    for c = 1:num_labels,
        y_temp(c,1) = (y(i,:) == c);       
    end

%calculate sum for all the k nodes in each data set
for k = 1 : num_labels,
    sum = sum + ((y_temp(k,:) * log(hx(:,k))) + ((1 - y_temp(k,:)) * log(1 - hx(:,k))));
end
J = J + sum;
end

J = (-1 * J)/m;

%------------------------Completed-----------------------

%------------------------Regularized Cost Function-----------------------

r_theta1 = size(Theta1, 1);
c_theta1 = size(Theta1, 2);

r_theta2 = size(Theta2, 1);
c_theta2 = size(Theta2, 2);

sum_theta1 = 0;
for i = 1 : r_theta1,
    temp_theta1 = Theta1(i,:);
    temp_sum = 0;
    for j = 2 : c_theta1,
        temp_sum = temp_sum + (temp_theta1(:,j) .^ 2); 
    end
    sum_theta1 = sum_theta1 + temp_sum;
end

sum_theta2 = 0;
for i = 1 : r_theta2,
    temp_theta2 = Theta2(i,:);
    temp_sum = 0;
    for j = 2 : c_theta2,
        temp_sum = temp_sum + (temp_theta2(:,j) .^ 2); 
    end
    sum_theta2 = sum_theta2 + temp_sum;
end


J = J + ((lambda * (sum_theta1 + sum_theta2))/(2 * m));

%------------------------Completed-----------------------

%------------------------Backpropagation Algorithm-----------------------
delta2 = 0;
delta1 = 0;
for i = 1 : m,
    
    for c = 1:num_labels,
        y_temp(c,1) = (y(i,:) == c);       
    end

    d_layer3 = a_layer3(i,:) - y_temp';    %d3 with dimension 1*10[a3 is 1*10 and y' is 1 * 10]

    d_layer2 = ((Theta2)' * d_layer3') .* a_layer2(i,:)'.* (1- a_layer2(i,:))';  
                                %d2 with dimension 26*1[Theta2' is 26*10, d3' is 10*1,
                                %a2' is 26*1 and (1-a2)' is 26 *1]

    d_layer2 = d_layer2(2:end,:);  %d2 dimension is 25*1
    

    %Accumulate the gradients 
    delta2 = delta2 + d_layer3' * a_layer2(i,:);   %delta2 is 10*26 [d3' is 10*1 and a2 is 1*26] 
    delta1 = delta1 + d_layer2 * X(i,:);  %delta1 is 25*401 [d2 is 25*1 and X is 1*401]

end


%------------------------Regularized Theta-----------------------

if (lambda == 0),
    Theta2_grad = (delta2/m);
    Theta1_grad = (delta1/m);
else,
    Temp_Theta2 = Theta2;
    Temp_Theta2(:,2:end) = lambda * Temp_Theta2(:,2:end);
    Temp_Theta2(:,1) = 0 .* Temp_Theta2(:,1);
    
    Temp_Theta1 = Theta1;
    Temp_Theta1(:,2:end) = lambda * Temp_Theta1(:,2:end);
    Temp_Theta1(:,1) = 0 .* Temp_Theta1(:,1);
    
    Theta2_grad = Theta2_grad + (delta2 + Temp_Theta2)/m;
    Theta1_grad = Theta1_grad + (delta1 + Temp_Theta1)/m;
end
 
%------------------------Completed-----------------------

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
