function [Phi,om] = getModalParameter(fulleqm,num)

[Phi,om] = eigs(fulleqm.K_sum,fulleqm.M_sum,num,eps); % eps �����ݲ� 
 % ���ضԽǾ��� om�;��� Phi��om�������Խ����ϵ�����ֵ�� num ��ģ��������ֵ����Phi�ĸ����а�����Ӧ������������
 
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