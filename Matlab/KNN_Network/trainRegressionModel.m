function [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
% [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
% 返回经过训练的回归模型及其 RMSE。以下代码重新创建在 Regression Learner App 中训练的
% 模型。您可以使用该生成的代码基于新数据自动训练同一模型，或通过它了解如何以程序化方式训练模
% 型。
%
%  输入:
%      trainingData: 一个所含预测变量和响应列与导入 App 中的相同的表。
%
%  输出:
%      trainedModel: 一个包含训练的回归模型的结构体。该结构体中具有各种关于所训练模型的
%       信息的字段。
%
%      trainedModel.predictFcn: 一个对新数据进行预测的函数。
%
%      validationRMSE: 一个包含 RMSE 的双精度值。在 App 中，"历史记录" 列表显示每个
%       模型的 RMSE。
%
% 使用该代码基于新数据来训练模型。要重新训练模型，请使用原始数据或新数据作为输入参数
% trainingData 从命令行调用该函数。
%
% 例如，要重新训练基于原始数据集 T 训练的回归模型，请输入:
%   [trainedModel, validationRMSE] = trainRegressionModel(T)
%
% 要使用返回的 "trainedModel" 对新数据 T2 进行预测，请使用
%   yfit = trainedModel.predictFcn(T2)
%
% T2 必须是一个表，其中至少包含与训练期间使用的预测变量列相同的预测变量列。有关详细信息，请
% 输入:
%   trainedModel.HowToPredict

% 由 MATLAB 于 2020-05-10 00:47:00 自动生成


% 提取预测变量和响应
% 以下代码将数据处理为合适的形状以训练模型。
%
inputTable = trainingData;
predictorNames = {'m2', 'm3', 'm4', 'k', 'alpha', 'beta'};
predictors = inputTable(:, predictorNames);
response = inputTable.EVnorm3_3;
isCategoricalPredictor = [false, false, false, false, false, false];

% 将 PCA 应用于预测变量矩阵。
% 仅对数值预测变量运行 PCA。PCA 不会对分类预测变量进行任何处理。
isCategoricalPredictorBeforePCA = isCategoricalPredictor;
numericPredictors = predictors(:, ~isCategoricalPredictor);
numericPredictors = table2array(varfun(@double, numericPredictors));
% 在 PCA 中必须将 'inf' 值视为缺失数据。
numericPredictors(isinf(numericPredictors)) = NaN;
[pcaCoefficients, pcaScores, ~, ~, explained, pcaCenters] = pca(...
    numericPredictors);
% 保留足够的成分来解释所需的方差量。
explainedVarianceToKeepAsFraction = 95/100;
numComponentsToKeep = find(cumsum(explained)/sum(explained) >= explainedVarianceToKeepAsFraction, 1);
pcaCoefficients = pcaCoefficients(:,1:numComponentsToKeep);
predictors = [array2table(pcaScores(:,1:numComponentsToKeep)), predictors(:, isCategoricalPredictor)];
isCategoricalPredictor = [false(1,numComponentsToKeep), true(1,sum(isCategoricalPredictor))];

% 训练回归模型
% 以下代码指定所有模型选项并训练模型。
concatenatedPredictorsAndResponse = predictors;
concatenatedPredictorsAndResponse.EVnorm3_3 = response;
linearModel = fitlm(...
    concatenatedPredictorsAndResponse, ...
    'linear', ...
    'RobustOpts', 'off');

% 使用预测函数创建结果结构体
predictorExtractionFcn = @(t) t(:, predictorNames);
pcaTransformationFcn = @(x) [ array2table((table2array(varfun(@double, x(:, ~isCategoricalPredictorBeforePCA))) - pcaCenters) * pcaCoefficients), x(:,isCategoricalPredictorBeforePCA) ];
linearModelPredictFcn = @(x) predict(linearModel, x);
trainedModel.predictFcn = @(x) linearModelPredictFcn(pcaTransformationFcn(predictorExtractionFcn(x)));

% 向结果结构体中添加字段
trainedModel.RequiredVariables = {'alpha', 'beta', 'k', 'm2', 'm3', 'm4'};
trainedModel.PCACenters = pcaCenters;
trainedModel.PCACoefficients = pcaCoefficients;
trainedModel.LinearModel = linearModel;
trainedModel.About = '此结构体是从 Regression Learner R2019b 导出的训练模型。';
trainedModel.HowToPredict = sprintf('要对新表 T 进行预测，请使用: \n yfit = c.predictFcn(T) \n将 ''c'' 替换为作为此结构体的变量的名称，例如 ''trainedModel''。\n \n表 T 必须包含由以下内容返回的变量: \n c.RequiredVariables \n变量格式(例如矩阵/向量、数据类型)必须与原始训练数据匹配。\n忽略其他变量。\n \n有关详细信息，请参阅 <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>。');

% 提取预测变量和响应
% 以下代码将数据处理为合适的形状以训练模型。
%
inputTable = trainingData;
predictorNames = {'m2', 'm3', 'm4', 'k', 'alpha', 'beta'};
predictors = inputTable(:, predictorNames);
response = inputTable.EVnorm3_3;
isCategoricalPredictor = [false, false, false, false, false, false];

% 执行交叉验证
KFolds = 5;
cvp = cvpartition(size(response, 1), 'KFold', KFolds);
% 将预测初始化为适当的大小
validationPredictions = response;
for fold = 1:KFolds
    trainingPredictors = predictors(cvp.training(fold), :);
    trainingResponse = response(cvp.training(fold), :);
    foldIsCategoricalPredictor = isCategoricalPredictor;
    
    % 将 PCA 应用于预测变量矩阵。
    % 仅对数值预测变量运行 PCA。PCA 不会对分类预测变量进行任何处理。
    isCategoricalPredictorBeforePCA = foldIsCategoricalPredictor;
    numericPredictors = trainingPredictors(:, ~foldIsCategoricalPredictor);
    numericPredictors = table2array(varfun(@double, numericPredictors));
    % 在 PCA 中必须将 'inf' 值视为缺失数据。
    numericPredictors(isinf(numericPredictors)) = NaN;
    [pcaCoefficients, pcaScores, ~, ~, explained, pcaCenters] = pca(...
        numericPredictors);
    % 保留足够的成分来解释所需的方差量。
    explainedVarianceToKeepAsFraction = 95/100;
    numComponentsToKeep = find(cumsum(explained)/sum(explained) >= explainedVarianceToKeepAsFraction, 1);
    pcaCoefficients = pcaCoefficients(:,1:numComponentsToKeep);
    trainingPredictors = [array2table(pcaScores(:,1:numComponentsToKeep)), trainingPredictors(:, foldIsCategoricalPredictor)];
    foldIsCategoricalPredictor = [false(1,numComponentsToKeep), true(1,sum(foldIsCategoricalPredictor))];
    
    % 训练回归模型
    % 以下代码指定所有模型选项并训练模型。
    concatenatedPredictorsAndResponse = trainingPredictors;
    concatenatedPredictorsAndResponse.EVnorm3_3 = trainingResponse;
    linearModel = fitlm(...
        concatenatedPredictorsAndResponse, ...
        'linear', ...
        'RobustOpts', 'off');
    
    % 使用预测函数创建结果结构体
    pcaTransformationFcn = @(x) [ array2table((table2array(varfun(@double, x(:, ~isCategoricalPredictorBeforePCA))) - pcaCenters) * pcaCoefficients), x(:,isCategoricalPredictorBeforePCA) ];
    linearModelPredictFcn = @(x) predict(linearModel, x);
    validationPredictFcn = @(x) linearModelPredictFcn(pcaTransformationFcn(x));
    
    % 向结果结构体中添加字段
    
    % 计算验证预测
    validationPredictors = predictors(cvp.test(fold), :);
    foldPredictions = validationPredictFcn(validationPredictors);
    
    % 按原始顺序存储预测
    validationPredictions(cvp.test(fold), :) = foldPredictions;
end

% 计算验证 RMSE
isNotMissing = ~isnan(validationPredictions) & ~isnan(response);
validationRMSE = sqrt(nansum(( validationPredictions - response ).^2) / numel(response(isNotMissing) ));
