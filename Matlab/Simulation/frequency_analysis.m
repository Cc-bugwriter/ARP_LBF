function [f,psd,octaves,rms_octaves] = frequency_analysis(x,Fs,Nfft,window,t_start)

%% Output Values
% f: frequencies of psd
% psd: power spectral density for frequencies f
% octaves: frequencies of octaves
% rms_octaves: RMS of octaves

% x: analyzed signal
% Fs: Sampling Frequency
% Nfft: Number of analyzed elements, best use with Nfft is of type 2^n
% window: set 'hann' for van Hann window, otherwise rectangular window
% t_start: Start time, make sure Fs*t_start+Nfft does not exceed size of x

% Frequencies of Octave Band
oktavmitte1 = 1; % Center Frequency of first octave
oktavzahl = 6;   % number of octaves
oktavmitten = oktavmitte1 * 2.^(0:(oktavzahl-1));
octaves = oktavmitten;

% Calculate Start Point
t_start=t_start*Fs+1;

% Define Window
if  strcmp(window,'hann')
    w1 = 2*sqrt(2/3)*hann(Nfft); %see : Digitale Signalverarbeitung 1, p.194
else
    w1 = ones(Nfft,1);
end

% Read Time Window
x_ausw=x(t_start:(t_start+Nfft-1))';

% FFT of Time Window
try
    fft_x=fft(x_ausw.*w1,Nfft)/Nfft;
catch
    fft_x=fft(x_ausw.*w1',Nfft)/Nfft;
end

% Calculate PSD bnased on FFT
% see Kammeyer, Digitale Signalverarbeitung

f=Fs/2*linspace(0,1,Nfft/2+1);
df=f(2)-f(1);
psd = (1/df)*2*(fft_x.*conj(fft_x));
psd = psd(1:Nfft/2+1);

% Calculate Power in Octave Band
oktavleistung = zeros(1,length(oktavmitten));

% Identify Frequencies
n_f = 50;                               % number of frequencies per octave
i_f = 0:n_f*(length(oktavmitten)+1);    % Index of frequencies

f_i = oktavmitten(1) * 2.^((i_f-n_f)/n_f);  % Calculate Frequencies for Octave Band Analysis
dfi = f_i * (2.^(0.5/n_f)-2.^(-0.5/n_f));   % Calculate Difference
window = 0.5*(cos(log2(f_i(1:(2*n_f+1)))*pi)+1); % Weighting Function for overlapping double-octaves

% Interpolation of PSD
psdIp = interp1(f,psd,f_i,'pchip');

% Set Borders
ende=find(f_i<72);
ende=ende(end);
psdIp=psdIp(1:ende);

% Weighting of analyzed window, Calculate Power
for i=0:length(oktavmitten)-1
    psd_ausw = psdIp((n_f*i+1):(n_f*i+2*n_f+1));
    psd_hann = psd_ausw.*window;
    oktavleistung(i+1) = sum(psd_hann.*(dfi((n_f*i+1):(n_f*i+2*n_f+1))));    
end

% Calculate RMS based on power
rms_octaves=sqrt(oktavleistung);

