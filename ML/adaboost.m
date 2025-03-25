%% Adaptive Boosting
function prediction = adaboost(feat, label, opts)

if isfield(opts,'Model'), Model = opts.Model; end
t = templateTree('MaxNumSplits', 1); % Weak learner template (decision tree with max depth 1)
numTrees = 50;

trainIdx = Model.training;    testIdx = Model.test;
xtrain   = feat(trainIdx,:);  ytrain  = label(trainIdx);
xvalid   = feat(testIdx,:);   yvalid  = label(testIdx);
% Training model

My_Model = fitensemble(xtrain, ytrain, 'AdaBoostM1', numTrees, t);
% Prediction
[pred, pred_prob]     = predict(My_Model, xvalid);

prediction.class_names   = My_Model.ClassNames;
prediction.pred         = pred;
prediction.label        = yvalid;
prediction.pred_prob    = pred_prob;
end
