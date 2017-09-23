function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
y1=y';
costVector = -(y1*log((sigmoid(theta'*X'))') + (ones(size(y1))-y1)*log(ones(size(X))-(sigmoid(theta'*X'))'));
J = costVector(1)/m;


gradNew = grad';
for i = 1:length(gradNew)
gradNew(i) = 0;
 for j = 1:m
  gradNew(i) =gradNew(i) + (sigmoid(theta'*X'(:,j))-y1(j))*X'(i,j);
 end
 gradNew(i) = gradNew(i)/m;
 end
 
 grad = gradNew';





% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
