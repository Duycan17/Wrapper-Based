% K-nearest Neighbor (9/12/2020)

function prediction = jknn(feat,label,opts)
% Default of k-value
k = 5;

if isfield(opts,'k'), k = opts.k; end
if isfield(opts,'Model'), Model = opts.Model; end

% Define training & validation sets
trainIdx = Model.training;    testIdx = Model.test;
xtrain   = feat(trainIdx,:);  ytrain  = label(trainIdx);
xvalid   = feat(testIdx,:);   yvalid  = label(testIdx);
% Training model
My_Model = fitcknn(xtrain, ytrain, 'NumNeighbors', k); 

% Prediction
[pred, pred_prob]     = predict(My_Model, xvalid);

prediction.class_names   = My_Model.ClassNames;
prediction.pred         = pred;
prediction.label        = yvalid;
prediction.pred_prob    = pred_prob;
end




