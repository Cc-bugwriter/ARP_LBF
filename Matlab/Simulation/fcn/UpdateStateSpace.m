function [ssmodal] = UpdateStateSpace(x,Theta,fulleqm,T)

% Verändern der Systematrizen 
[fulleqm] = MKUpdate(x,Theta,fulleqm);
%% Zusammensetzen der Submatrizen
fulleqm.K_sum = sparse(size(fulleqm.K{1},1),size(fulleqm.K{1},1));
for SUM = 1:length(fulleqm.K)
    fulleqm.K_sum = fulleqm.K_sum + fulleqm.K{SUM};
end
fulleqm.M_sum = sparse(size(fulleqm.M{1},1),size(fulleqm.M{1},1));
for SUM = 1:length(fulleqm.M)
    fulleqm.M_sum = fulleqm.M_sum + fulleqm.M{SUM};
end
% Damping (Rayleigh Damping) 
alpha   = 0.6261;
beta    = 0.0001;
fulleqm.D = alpha*fulleqm.M_sum+beta*fulleqm.K_sum;
%% Bestimmen Zustandsraumdarstellung
[ssmodal] = getModalStateSpaceDerivative(fulleqm,T);

return