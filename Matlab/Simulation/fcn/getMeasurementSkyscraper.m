function [Mea] = getMeasurementSkyscraper(x,fulleqm,num_Mea,num_EIG,Theta)


mea_noise = normrnd(0,0.005,[num_Mea,1]);
[fulleqm] = MKUpdate(x,Theta,fulleqm);
%% Zusammensetzen Submatrizen
fulleqm.K_sum = sparse(size(fulleqm.K{1},1),size(fulleqm.K{1},1));
for SUM = 1:length(fulleqm.K)
    fulleqm.K_sum = fulleqm.K_sum + fulleqm.K{SUM};
end
fulleqm.M_sum = sparse(size(fulleqm.M{1},1),size(fulleqm.M{1},1));
for SUM = 1:length(fulleqm.M)
    fulleqm.M_sum = fulleqm.M_sum + fulleqm.M{SUM};
end
%% Bestimmen Eigenwerte
[~,D] = eigs(fulleqm.K_sum,fulleqm.M_sum,num_EIG,eps);
for  ii = 1:num_Mea
    Mea.EV(:,ii) = diag(D)+mea_noise(ii).*diag(D);
end
Mea.eps_var = var(Mea.EV,0,2)./mean(Mea.EV,2);
return