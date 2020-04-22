function [fulleqm] = MKUpdate(x,Theta,fulleqm)
%
for p = 1:size(x,2)
    if Theta.Name{p} == 'E'
        %
        fulleqm.K{Theta.No(p)} =fulleqm.K{Theta.No(p)}*x(p);
        %
    elseif Theta.Name{p} == 'h' % “‚“Â£ø
        fulleqm.K{Theta.No(p)} = fulleqm.K{Theta.No(p)}*(x(p))^3;
        fulleqm.M{Theta.No(p)} = fulleqm.M{Theta.No(p)}*x(p);
    elseif Theta.Name{p} == 'rho'
        fulleqm.M{Theta.No(p)} = fulleqm.M{Theta.No(p)}*x(p);
    end
end
end