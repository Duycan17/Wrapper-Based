%% Random Forest

function prediction = rf(feat, label, opts)

if isfield(opts,'Model'), Model = opts.Model; end

% Define training & validation sets
trainIdx = Model.training;    testIdx = Model.test;
xtrain   = feat(trainIdx,:);  ytrain  = label(trainIdx);
xvalid   = feat(testIdx,:);   yvalid  = label(testIdx);
% Training model
My_Model = fitcensemble(xtrain, ytrain, "Method", "Bag"); 

% Prediction
[pred, pred_prob]     = predict(My_Model, xvalid);

prediction.class_names   = My_Model.ClassNames;
prediction.pred         = pred;
prediction.label        = yvalid;
prediction.pred_prob    = pred_prob;