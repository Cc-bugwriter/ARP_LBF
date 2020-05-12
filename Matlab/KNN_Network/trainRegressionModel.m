function [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
% [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
% ���ؾ���ѵ���Ļع�ģ�ͼ��� RMSE�����´������´����� Regression Learner App ��ѵ����
% ģ�͡�������ʹ�ø����ɵĴ�������������Զ�ѵ��ͬһģ�ͣ���ͨ�����˽�����Գ��򻯷�ʽѵ��ģ
% �͡�
%
%  ����:
%      trainingData: һ������Ԥ���������Ӧ���뵼�� App �е���ͬ�ı�
%
%  ���:
%      trainedModel: һ������ѵ���Ļع�ģ�͵Ľṹ�塣�ýṹ���о��и��ֹ�����ѵ��ģ�͵�
%       ��Ϣ���ֶΡ�
%
%      trainedModel.predictFcn: һ���������ݽ���Ԥ��ĺ�����
%
%      validationRMSE: һ������ RMSE ��˫����ֵ���� App �У�"��ʷ��¼" �б���ʾÿ��
%       ģ�͵� RMSE��
%
% ʹ�øô��������������ѵ��ģ�͡�Ҫ����ѵ��ģ�ͣ���ʹ��ԭʼ���ݻ���������Ϊ�������
% trainingData �������е��øú�����
%
% ���磬Ҫ����ѵ������ԭʼ���ݼ� T ѵ���Ļع�ģ�ͣ�������:
%   [trainedModel, validationRMSE] = trainRegressionModel(T)
%
% Ҫʹ�÷��ص� "trainedModel" �������� T2 ����Ԥ�⣬��ʹ��
%   yfit = trainedModel.predictFcn(T2)
%
% T2 ������һ�����������ٰ�����ѵ���ڼ�ʹ�õ�Ԥ���������ͬ��Ԥ������С��й���ϸ��Ϣ����
% ����:
%   trainedModel.HowToPredict

% �� MATLAB �� 2020-05-10 00:47:00 �Զ�����


% ��ȡԤ���������Ӧ
% ���´��뽫���ݴ���Ϊ���ʵ���״��ѵ��ģ�͡�
%
inputTable = trainingData;
predictorNames = {'m2', 'm3', 'm4', 'k', 'alpha', 'beta'};
predictors = inputTable(:, predictorNames);
response = inputTable.EVnorm3_3;
isCategoricalPredictor = [false, false, false, false, false, false];

% �� PCA Ӧ����Ԥ���������
% ������ֵԤ��������� PCA��PCA ����Է���Ԥ����������κδ���
isCategoricalPredictorBeforePCA = isCategoricalPredictor;
numericPredictors = predictors(:, ~isCategoricalPredictor);
numericPredictors = table2array(varfun(@double, numericPredictors));
% �� PCA �б��뽫 'inf' ֵ��Ϊȱʧ���ݡ�
numericPredictors(isinf(numericPredictors)) = NaN;
[pcaCoefficients, pcaScores, ~, ~, explained, pcaCenters] = pca(...
    numericPredictors);
% �����㹻�ĳɷ�����������ķ�������
explainedVarianceToKeepAsFraction = 95/100;
numComponentsToKeep = find(cumsum(explained)/sum(explained) >= explainedVarianceToKeepAsFraction, 1);
pcaCoefficients = pcaCoefficients(:,1:numComponentsToKeep);
predictors = [array2table(pcaScores(:,1:numComponentsToKeep)), predictors(:, isCategoricalPredictor)];
isCategoricalPredictor = [false(1,numComponentsToKeep), true(1,sum(isCategoricalPredictor))];

% ѵ���ع�ģ��
% ���´���ָ������ģ��ѡ�ѵ��ģ�͡�
concatenatedPredictorsAndResponse = predictors;
concatenatedPredictorsAndResponse.EVnorm3_3 = response;
linearModel = fitlm(...
    concatenatedPredictorsAndResponse, ...
    'linear', ...
    'RobustOpts', 'off');

% ʹ��Ԥ�⺯����������ṹ��
predictorExtractionFcn = @(t) t(:, predictorNames);
pcaTransformationFcn = @(x) [ array2table((table2array(varfun(@double, x(:, ~isCategoricalPredictorBeforePCA))) - pcaCenters) * pcaCoefficients), x(:,isCategoricalPredictorBeforePCA) ];
linearModelPredictFcn = @(x) predict(linearModel, x);
trainedModel.predictFcn = @(x) linearModelPredictFcn(pcaTransformationFcn(predictorExtractionFcn(x)));

% �����ṹ��������ֶ�
trainedModel.RequiredVariables = {'alpha', 'beta', 'k', 'm2', 'm3', 'm4'};
trainedModel.PCACenters = pcaCenters;
trainedModel.PCACoefficients = pcaCoefficients;
trainedModel.LinearModel = linearModel;
trainedModel.About = '�˽ṹ���Ǵ� Regression Learner R2019b ������ѵ��ģ�͡�';
trainedModel.HowToPredict = sprintf('Ҫ���±� T ����Ԥ�⣬��ʹ��: \n yfit = c.predictFcn(T) \n�� ''c'' �滻Ϊ��Ϊ�˽ṹ��ı��������ƣ����� ''trainedModel''��\n \n�� T ����������������ݷ��صı���: \n c.RequiredVariables \n������ʽ(�������/��������������)������ԭʼѵ������ƥ�䡣\n��������������\n \n�й���ϸ��Ϣ������� <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>��');

% ��ȡԤ���������Ӧ
% ���´��뽫���ݴ���Ϊ���ʵ���״��ѵ��ģ�͡�
%
inputTable = trainingData;
predictorNames = {'m2', 'm3', 'm4', 'k', 'alpha', 'beta'};
predictors = inputTable(:, predictorNames);
response = inputTable.EVnorm3_3;
isCategoricalPredictor = [false, false, false, false, false, false];

% ִ�н�����֤
KFolds = 5;
cvp = cvpartition(size(response, 1), 'KFold', KFolds);
% ��Ԥ���ʼ��Ϊ�ʵ��Ĵ�С
validationPredictions = response;
for fold = 1:KFolds
    trainingPredictors = predictors(cvp.training(fold), :);
    trainingResponse = response(cvp.training(fold), :);
    foldIsCategoricalPredictor = isCategoricalPredictor;
    
    % �� PCA Ӧ����Ԥ���������
    % ������ֵԤ��������� PCA��PCA ����Է���Ԥ����������κδ���
    isCategoricalPredictorBeforePCA = foldIsCategoricalPredictor;
    numericPredictors = trainingPredictors(:, ~foldIsCategoricalPredictor);
    numericPredictors = table2array(varfun(@double, numericPredictors));
    % �� PCA �б��뽫 'inf' ֵ��Ϊȱʧ���ݡ�
    numericPredictors(isinf(numericPredictors)) = NaN;
    [pcaCoefficients, pcaScores, ~, ~, explained, pcaCenters] = pca(...
        numericPredictors);
    % �����㹻�ĳɷ�����������ķ�������
    explainedVarianceToKeepAsFraction = 95/100;
    numComponentsToKeep = find(cumsum(explained)/sum(explained) >= explainedVarianceToKeepAsFraction, 1);
    pcaCoefficients = pcaCoefficients(:,1:numComponentsToKeep);
    trainingPredictors = [array2table(pcaScores(:,1:numComponentsToKeep)), trainingPredictors(:, foldIsCategoricalPredictor)];
    foldIsCategoricalPredictor = [false(1,numComponentsToKeep), true(1,sum(foldIsCategoricalPredictor))];
    
    % ѵ���ع�ģ��
    % ���´���ָ������ģ��ѡ�ѵ��ģ�͡�
    concatenatedPredictorsAndResponse = trainingPredictors;
    concatenatedPredictorsAndResponse.EVnorm3_3 = trainingResponse;
    linearModel = fitlm(...
        concatenatedPredictorsAndResponse, ...
        'linear', ...
        'RobustOpts', 'off');
    
    % ʹ��Ԥ�⺯����������ṹ��
    pcaTransformationFcn = @(x) [ array2table((table2array(varfun(@double, x(:, ~isCategoricalPredictorBeforePCA))) - pcaCenters) * pcaCoefficients), x(:,isCategoricalPredictorBeforePCA) ];
    linearModelPredictFcn = @(x) predict(linearModel, x);
    validationPredictFcn = @(x) linearModelPredictFcn(pcaTransformationFcn(x));
    
    % �����ṹ��������ֶ�
    
    % ������֤Ԥ��
    validationPredictors = predictors(cvp.test(fold), :);
    foldPredictions = validationPredictFcn(validationPredictors);
    
    % ��ԭʼ˳��洢Ԥ��
    validationPredictions(cvp.test(fold), :) = foldPredictions;
end

% ������֤ RMSE
isNotMissing = ~isnan(validationPredictions) & ~isnan(response);
validationRMSE = sqrt(nansum(( validationPredictions - response ).^2) / numel(response(isNotMissing) ));
