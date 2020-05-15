function [a] = ReLU(z)
%ReLU function for acitvation neuron
%   z: intermediate variable in neuron
%   a: acitvation output
a = zeros(1, length(z));
for i = 1:length(z)
    if z(i) >=  0
        a(i) = z(i);
    else
        a(i) = 0;
    end
end
end

