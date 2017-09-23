function [J, grad] = costFunctionLogisticRegression(theta, X, y, lambda)
% number of training examples
n = length(y); 

% pre-allocate space for gradient
grad = zeros(size(theta));

% Logistic Regression Cost Function
J = (1/n)*sum(-y.*(log(sigmoid(X*theta)))-(1-y).*log(1-(sigmoid(X*theta))))+(lambda/(2*n))*sum(theta(2:end).^2);

grad = (1/n)*(X'*(sigmoid(X*theta)-y))+(theta.*lambda)/n;
grad(1) = (1/n)*sum((sigmoid(X*theta)-y).*X(:,1));

end
