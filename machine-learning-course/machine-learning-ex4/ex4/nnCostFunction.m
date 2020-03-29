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

new_y = zeros(rows(y), num_labels);
for i = 1:rows(y)
    switch y(i)
        case 1
            new_y(i,1) = 1;
        case 2
            new_y(i,2) = 1;
        case 3
            new_y(i,3) = 1;
        case 4
            new_y(i,4) = 1;
        case 5
            new_y(i,5) = 1;
        case 6
            new_y(i,6) = 1;
        case 7
            new_y(i,7) = 1;
        case 8
            new_y(i,8) = 1;
        case 9
            new_y(i,9) = 1;
        case 0
            new_y(i,10) = 1;
    end
end

t= sigmoid([ones(m, 1) X] * Theta1');
h = sigmoid([ones(size(t,1), 1) t] * Theta2');

for k = 1:num_labels
    J = J + sum(-new_y(:,k) .* log(h(:,k)) - (1-new_y(:,k)) .* log(1-h(:,k)));
end

J = J / m;

sum_theta1 = 0;
sum_theta2 = 0;

for k = 2:columns(Theta1)
    for j = 1:rows(Theta1)
        sum_theta1 = sum_theta1 + Theta1(j,k)^2;
    end
end

for k = 2:columns(Theta2)
    for j = 1:rows(Theta2)
        sum_theta2 = sum_theta2 + Theta2(j,k)^2;
    end
end

J = J + lambda * (sum_theta1 + sum_theta2) / (2 * m);

X = [ones(size(X, 1), 1) X];
a1 = zeros(rows(X), columns(X));

for i = 1:m
    a1(i,:) = X(i,:);
    %a1_1 = [ones(size(a1, 1), 1) a1];
    z2(i,:) = Theta1 * a1(i,:)';
    a2_1(i,:) = sigmoid(z2(i,:));
    a2(i,:) = [1 a2_1(i,:)];
    z3(i,:) = Theta2 * a2(i,:)';
    a3(i,:) = sigmoid(z3(i,:));
end

%a2
%a3

for k = 1:num_labels
    d3(:, k) = a3(:, k) - (y == k); 
end

%d3

%size(a3)
%size(d3)
%size(Theta2(:,2:end))
%size(z2)

d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2);

Delta1 = a1' * d2;
Delta1 = Delta1'; 

Delta2 = a2' * d3;
Delta2 = Delta2';

%size(Delta1)
%size(Delta2)

Theta1_grad = Delta1 / m;
Theta2_grad = Delta2 / m;

reg1 = 0;
for i = 1:rows(Theta1)
    for j = 2:columns(Theta1)
        reg1 = reg1 + Theta1(i,j);
    end
end
reg1 = lambda * reg1 / m;


reg2 = 0;
for i = 1:rows(Theta2)
    for j = 2:columns(Theta2)
        reg2 = reg2 + Theta2(i,j);
    end
end
reg2 = lambda * reg2 / m;

Theta1_temp = Theta1_grad(:,1);
Theta2_temp = Theta2_grad(:,1);
Theta1_grad = Theta1_grad + lambda * Theta1 / m;
Theta2_grad = Theta2_grad + lambda * Theta2 / m;
Theta1_grad = [Theta1_temp Theta1_grad(:,2:end)];
Theta2_grad = [Theta2_temp Theta2_grad(:,2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
