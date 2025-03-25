%% Extra Tree
function prediction = extratree(feat, label, opts)

if isfield(opts,'Model'), Model = opts.Model; end
numTrees = 50;
local.opts = statset('UseParallel',true);

trainIdx = Model.training;    testIdx = Model.test;
xtrain   = feat(trainIdx,:);  ytrain  = label(trainIdx);
xvalid   = feat(testIdx,:);   yvalid  = label(testIdx);
% Training model

My_Model = TreeBagger(numTrees, xtrain, ytrain, ...
    'Method', 'classification', ... % 'classification' or 'regression'
    'NumPredictorsToSample', 'all', ... % Use all predictors at each split
    'MinLeafSize', 1, ... % Minimum number of observations per tree leaf
    'Options', local.opts); 

% Prediction
[pred, pred_prob]     = predict(My_Model, xvalid);

prediction.class_names   = My_Model.ClassNames;
prediction.pred         = cellfun(@str2double, pred);
prediction.label        = yvalid;
prediction.pred_prob    = pred_prob;
end
