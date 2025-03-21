function metrics = evaluate(prediction)
% metrics
pred        = prediction.pred.';
label       = prediction.label.';
pred_prob   = prediction.pred_prob;
classNames  = prediction.class_names;

% confusion matrix
[m, order]  = confusionmat(label, pred);
sum_rows    = sum(m, 2);
sum_col     = sum(m, 1);
diagonal    = diag(m);

% precision
precision   = mean(diagonal./sum_rows);
% recall 
recall      = mean(diagonal./sum_col');
% acc 
acc         = sum(pred == label) / length(label);
% f1 
f1          = 2 * ((precision * recall)/(precision + recall));
% auc 
auc         = rocmetrics(label, pred_prob, classNames).AUC(1);

metrics.acc = acc;
metrics.recall = recall;
metrics.f1 = f1;
metrics.auc = auc;
metrics.precision = precision;

fprintf('\n Accuracy: %g %%', 100 * acc);

fprintf('\n auc: %g ', auc);

fprintf('\n f1: %g ', f1);

fprintf('\n Precision: %g ', precision);

fprintf('\n Recall: %g \n', recall);

fprintf('\n');

end