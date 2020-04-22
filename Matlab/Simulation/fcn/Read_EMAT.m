function [M,K,Elm] = Read_EMAT(RST,EMAT,NUM_ELEMENT)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
elmData         = readElements(RST);
clear RST  
Elm.elm         = EMAT.ematInfo.nume;       % Number of Elements
Elm.node        = EMAT.ematInfo.lenbac;     % Number of Nodes
Elm.dof_total   = EMAT.ematInfo.lenu;

% Initialisation Element Matrices
k = cell(1,Elm.elm);
m = cell(1,Elm.elm);
DOF_IndexTable = cell(1,Elm.elm);
DOF = cell(1,Elm.elm);
NODE = cell(1,Elm.elm);
contact = zeros(1,Elm.elm);
elm_typ = zeros(1,Elm.elm);
elm_node = zeros(1,Elm.elm);
elm_dof = zeros(1,Elm.elm);
% 
for ii = 1:Elm.elm
    elm_typ(ii) = elmData.elm{ii,1}(1,1);                   % Element 
    elm_node(ii)= length(unique(EMAT.elements{ii,1}.NOD));
    elm_dof(ii) = length(unique(EMAT.elements{ii,1}.DOF));    
    if  EMAT.elements{ii, 1}.header.stkey~=0 && EMAT.elements{ii, 1}.header.mkey~=0 
        k{ii} = EMAT.elements{ii, 1}.stf;
        m{ii} = EMAT.elements{ii,1}.mas;
        DOF_IndexTable{1,ii} = EMAT.elements{ii,1}.DOFIndexTable;
        DOF{1,ii} = EMAT.elements{ii, 1}.DOF;
        NODE{1,ii} = EMAT.elements{ii, 1}.NOD;
    elseif EMAT.elements{ii, 1}.header.stkey~=0 && EMAT.elements{ii, 1}.header.mkey~=1
        m{ii} = 0;
        temp_DOF = find(EMAT.elements{ii,1}.DOFIndexTable ~=0);
        DOF_IndexTable{1,ii} = EMAT.elements{ii,1}.DOFIndexTable(temp_DOF,1);
        k{ii} = EMAT.elements{ii, 1}.stf(temp_DOF,temp_DOF);
        DOF{1,ii} = EMAT.elements{ii, 1}.DOF(temp_DOF,1);
        NODE{1,ii} = EMAT.elements{ii, 1}.NOD(temp_DOF,1);
        contact(ii) =1;
    else
        contact(ii) =2;
    end
end

Elm.elm_typ         = elm_typ;          % Elementtyp
Elm.DOF_IndexTable  = DOF_IndexTable;   % IndexTable
Elm.contact         = contact;          % Contact Elements
Elm.nodes           = elm_node;         % Number of Nodes per Element
Elm.DOF             = DOF;
Elm.NODE            = NODE;
t = find(EMAT.BIT ~=1);

NODE_MAPPING = zeros(numel(Elm.elm_typ),1);
for ii = 1:Elm.elm
    if isempty(Elm.DOF_IndexTable{1,ii})
    else
        NODE_MAPPING(Elm.DOF_IndexTable{1,ii},:) = Elm.NODE{1,ii};
    end
end

Elm.NODE_MAPPING = NODE_MAPPING;
NODE_MAPPING(t,:) = [];
Elm.NODE_BC = NODE_MAPPING; 
%% Initialisation System Matrices
M = cell(1,NUM_ELEMENT);
K = cell(1,NUM_ELEMENT+1);

for ii = 1:NUM_ELEMENT+1
    if ii > NUM_ELEMENT
        K{1,ii} = zeros(Elm.dof_total);
    else
        M{1,ii} = zeros(Elm.dof_total);
        K{1,ii} = zeros(Elm.dof_total);
    end
end

%%
for ii = 1:Elm.elm
    if Elm.contact(ii) ==2 
    elseif Elm.contact(ii) ==1
        K{1,NUM_ELEMENT}(Elm.DOF_IndexTable{:,ii},Elm.DOF_IndexTable{:,ii}) = K{1,NUM_ELEMENT}(Elm.DOF_IndexTable{:,ii},Elm.DOF_IndexTable{:,ii}) + k{ii};
    elseif Elm.elm_typ(ii) ~=NUM_ELEMENT+1
        K{1,Elm.elm_typ(ii)}(Elm.DOF_IndexTable{:,ii},Elm.DOF_IndexTable{:,ii}) = K{1,Elm.elm_typ(ii)}(Elm.DOF_IndexTable{:,ii},Elm.DOF_IndexTable{:,ii}) + k{ii};
        M{1,Elm.elm_typ(ii)}(Elm.DOF_IndexTable{:,ii},Elm.DOF_IndexTable{:,ii}) = M{1,Elm.elm_typ(ii)}(Elm.DOF_IndexTable{:,ii},Elm.DOF_IndexTable{:,ii}) + m{ii};
    end
end
%% Randbedingungen
for ii = 1:NUM_ELEMENT+1
    if ii > NUM_ELEMENT
        K{1,ii}(t,:) = []; K{1,ii}(:,t) = [];
    else
        K{1,ii}(t,:) = []; K{1,ii}(:,t) = [];
        M{1,ii}(t,:) = []; M{1,ii}(:,t) = [];
    end
end

end

