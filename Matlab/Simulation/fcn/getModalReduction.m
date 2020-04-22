function [fulleqm_r] = getModalReduction(fulleqm,T)

fulleqm_r.K = cell(1,length(fulleqm.K));
for ii = 1:length(fulleqm.K)
    fulleqm_r.K{ii} = T'*fulleqm.K{ii}*T;
end
fulleqm_r.M = cell(1,length(fulleqm.M));
for ii = 1:length(fulleqm.K)
    fulleqm_r.M{ii} = T'*fulleqm.M{ii}*T;
end

fulleqm_r.K_sum = sparse(size(fulleqm_r.K{1},1),size(fulleqm_r.K{1},1));
for SUM = 1:length(fulleqm.K)
    fulleqm_r.K_sum = fulleqm_r.K_sum + fulleqm_r.K{SUM};
end
fulleqm_r.M_sum = sparse(size(fulleqm_r.M{1},1),size(fulleqm_r.M{1},1));
for SUM = 1:length(fulleqm_r.M)
    fulleqm_r.M_sum = fulleqm_r.M_sum + fulleqm_r.M{SUM};
end

if isfield(fulleqm,'D_sum')
    fulleqm_r.D_sum = T'*fulleqm.D_sum*T;
else
    
end


fulleqm_r.B = T'*fulleqm.B;
fulleqm_r.C = fulleqm.C*T;
return