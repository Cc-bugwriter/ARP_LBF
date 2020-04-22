%% Simulink (muss Simulink Model oeffnen)
% Simulink Einstellungen
fs  = 2e3;      % Sample Frequenz
Sample_Time = 1/fs; % Sample Time
Time   = 6;       % Simulationszeit

% Berechnen modal reduzierte State-Space Darstellung
[ssmodal] = UpdateStateSpace(x,Theta,fulleqm,T);

modelname='Hochhaus_Simulink';

set_param(modelname,'SolverType','Fixed-step');
set_param(modelname,'FixedStep',num2str(1/fs));
set_param(modelname,'StartTime','0');
set_param(modelname,'StopTime',num2str(Time));

sim(modelname);

Simulation_output = ans;
%% Frequency Domain Data Analysis
Simulation_output.xpp_fft = fft(Simulation_output.xpp);

n_xpp = length(Simulation_output.xpp);          % number of samples
f_xpp = (0:n_xpp-1)*(fs/n_xpp);     % frequency range
Simulation_output.xpp_psd = abs(Simulation_output.xpp_fft).^2/n_xpp;    % power of the DFT
rel_xpp = 10e-6;   % DIN EN ISO 1683 Bezugswert fuer Pegel
Simulation_output.xpp_psd = 10*log10(Simulation_output.xpp_psd / rel_xpp^2);  % power of DFT in dB Einheit

semilogx(f_xpp*(2*pi), Simulation_output.xpp_psd ,'linewidth',2 )
xlim([0 1500])
grid on
xlabel('Frequency Rad/s')
ylabel('Power in dB')
title('Frequency Domain Data Analysis')
hold on