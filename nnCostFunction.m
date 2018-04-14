function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

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


K = num_labels;
Y = eye(K)(y, :);

a1 = [ones(m, 1), X];

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1), a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

cost = sum((-Y .* log(a3)) - ((1 - Y) .* log(1 - a3)), 2);
J = (1 / m) * sum(cost);

Theta1NoBias = Theta1(:, 2:end);
Theta2NoBias = Theta2(:, 2:end);

reg  = (lambda / (2 * m)) * (sum(sumsq(Theta1NoBias)) + sum(sumsq(Theta2NoBias)));
J += reg;

Delta1 = 0;
Delta2 = 0;

for t = 1:m
	
	a1 = [1; X(t, :)']; 
	z2 = Theta1 * a1;
	a2 = [1; sigmoid(z2)]; 

	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

	
	d3 = a3 - Y(t, :)';
	
	
	d2 = (Theta2NoBias' * d3) .* sigmoidGradient(z2);

	
	Delta2 += (d3 * a2');
	Delta1 += (d2 * a1');
endfor


Theta1_grad = (1 / m) * Delta1;
Theta2_grad = (1 / m) * Delta2;

Theta1_grad(:, 2:end) += ((lambda / m) * Theta1NoBias);
Theta2_grad(:, 2:end) += ((lambda / m) * Theta2NoBias);

% -------------------------------------------------------------

% =========================================================================
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
