function [ssmodal] = getModalStateSpaceDerivative(fulleqm,Phi)

MD = -(Phi'*fulleqm.M_sum*Phi)\(Phi'*fulleqm.D_sum*Phi);
MD(abs(MD)<max(abs(MD(:)))*1e-12)=0;

MK = -(Phi'*fulleqm.M_sum*Phi)\(Phi'*fulleqm.K_sum*Phi);
MK(abs(MK)<max(abs(MK(:)))*1e-12)=0;

a = [MD,        MK ;...
speye(size(MK)), zeros(size(MK)) ]; % 状态空间里，以降序求导的形式分布状态量

b = (Phi'*fulleqm.M_sum*Phi)\Phi'*fulleqm.B; % 
b = [b;
zeros(size(b))];

c = fulleqm.C*Phi;
c = [c,   zeros(size(c))];

d =     zeros(size(c,1),size(b,2));

ssmodal.a = full(a);
ssmodal.b = full(b);
ssmodal.c = full(c);
ssmodal.d = d;
return