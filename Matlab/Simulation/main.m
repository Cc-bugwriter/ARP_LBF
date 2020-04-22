clear all
close all
clc
%% Funktionen hinzufügen
% addpath('C:\Users\huelsebrock\Documents\BMW_Promotion\MATLAB\Modellabgleich\00_fcn')
addpath('fcn\')
%% Daten laden

addpath('data\')
%     RST         = ma_ansys.result('file.rst');
%     EMAT        = emat('file.emat');
%     FULL        = ma_ansys.full('file.full');
%     [fulleqm,Elm]  = Read_EMAT_2(RST,EMAT,FULL);
%     load('RST.mat')
%     load('FULL.mat')
%     load('EMAT.mat')
    load('fulleqm.mat')
    load('Elm.mat')
    nodeIDs         =   [617, 1054, 838];           % Input-node IDs %FE-Modell, diskretisiertes Bauteil, Elemente an Knoten verbunden, Knotennummern, drei Knoten auf der Mitte der ersten drei Massen liegen. 
    xDir            =   [1, 0, 0, 0, 0, 0];         % X-Direction
    
    % Select Node
    selected_dofs   =   ismember(Elm.NODE_MAPPING,nodeIDs); %node mapping enthält Matrizen pro Knoten & Verbindung. wo ist Knoten 617: Elm.Node_mapping enthält 617 drei mal wegen 3 dof
    fulleqm.B       =   fulleqm.B(:,selected_dofs);
    fulleqm.C       =   fulleqm.C(selected_dofs,:);
    
    % Select Direction
    inChannels      =   zeros(sum(selected_dofs), 1);   % Bit-mask for input channels
    outChannels     =   zeros(sum(selected_dofs), 1);   % Bit-mask for output channels %Zustandsraummodell, einen Eingang an einem knoten, zwei Ausgänge an beiden anderen % virtual sensoring um dritten Knoten vorauszusagen -< Validierung
    
    fulleqm.inDir   =   Elm.inDir(selected_dofs,:);     % nur eine Richtung interessant, inDir Matriz ist volle Matrix, nur die Zeilen interessan (x,y,z von drei nodeIDs)
    fulleqm.outDir  =   Elm.outDir(selected_dofs,:);
    
    fulleqm.inChannels.node     = unique(Elm.NODE_MAPPING(selected_dofs),'stable'); %zeigt welcfher Eingang wo ist.
    fulleqm.outChannels.node    = unique(Elm.NODE_MAPPING(selected_dofs),'stable'); %jeder nodeID tritt nur noch einmal auf
    
    % Input Channels
    for m = 1:sum(selected_dofs)
        if isequal(fulleqm.inDir(m, :), xDir) %reduziert Matrix auf relevanten Richtungen
            inChannels(m) = 1;
        end
    end
    inChannels  =   logical(inChannels);
    fulleqm.B   = fulleqm.B(:,inChannels);
    
    % Output Channels    
    for m = 1:sum(selected_dofs)
        if isequal(fulleqm.inDir(m, :), xDir)
            outChannels(m) = 1;
        end
    end
    outChannels  =   logical(outChannels);
    fulleqm.C    =   fulleqm.C(outChannels,:);
    %% Modellreduktion
    num = 40;       % Number of Modes, eig  nur 3 relevant (3Massenschwinger
    [T,om] = getModalParameter(fulleqm,num); % T Transformationsmatrix, om Eigenfrequenzen
    
    % Damping (Rayleigh Damping)
    alpha       = 0.6261;
    beta        = 0.0001;
    fulleqm.D   = alpha*fulleqm.M_sum+beta*fulleqm.K_sum;
rmpath('data\')
%% Messungen
num_Mea     = 10;           % Anzahl Messungen
num_EIG     = 3;            % Anzahl gemessener Eigenfrequenzen, Eigenfrequenzen für unterschiedl Systemzustände berechnet
% Körper entsprechen der Nummerierung aus der Präsentation
Theta.No    = [2];          % Körper
% Mögliche Änderungen Dichte 'rho', Steifigkeit 'E', Höhe 'h' (nur für Element 5&6)
Theta.Name  = {'rho'};      % Parameter 
% Messungen 
[Mea] = getMeasurementSkyscraper(fulleqm,num_Mea,num_EIG,Theta);



