function [ j ] = costFunctionJ( x,y,theta )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

m = size(x,1);    %number of training examples
predictions = x*theta;   %predictions of hypothesis(h(x)) on all m examples
sqrdErrors = (predictions - y) .^ 2;   %squared differences

j = 1/(2*m) * sum(sqrdErrors);

end

