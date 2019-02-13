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
temp1=0;
temp2=0;
g=sigmoid(X*theta);
n=length(theta);
for i=1:m
  temp1=temp1+log(g(i))'*y(i)+log(1-g(i))'*(1-y(i));
end
for i=1:n
  temp2=temp2+theta(i);
end

J=-(temp1 - lambda*temp2)/m;

theta1 = [0 ; theta(2:size(theta), :)];
grad=(X'*(g-y)+lambda*theta1)/m;

% =============================================================

end
