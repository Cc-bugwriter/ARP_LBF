function [a] = sigmoid(z)
%Sigmoid function for acitvation neuron
%   z: intermediate variable in neuron
%   a: acitvation output
a = 1./(1+exp(-z));
end

