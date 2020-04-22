function [Phi,om] = getModalParameter(fulleqm,num)

[Phi,om] = eigs(fulleqm.K_sum,fulleqm.M_sum,num,eps); % eps 收敛容差 
 % 返回对角矩阵 om和矩阵 Phi，om包含主对角线上的特征值（ num 个模最大的特征值），Phi的各列中包含对应的特征向量。
 
% Nomalise
VtMV = Phi'*fulleqm.M_sum*Phi;
Phi = Phi*diag(sqrt(diag(VtMV).^(-1)));

% interf = any(fulleqm.B, 2) | any(fulleqm.C, 1)';
% 
% [EV, om] = eigs(fulleqm.K_sum(~interf,~interf), fulleqm.M_sum(~interf,~interf), num, 'smallestabs');
% 
% % fprintf(1, '%e \num', sort(sqrt(abs(diag(L)))/2/pi));
% 
% G_sm = -fulleqm.K_sum(~interf,~interf)\fulleqm.K_sum(~interf,interf);
% 
% Phi(interf,:)   = [speye(nnz(interf)) sparse(nnz(interf), num)];
% Phi(~interf,:)  = [G_sm sparse(EV)];
return