clear, clc, close
% Load data 
load AEEEM/modified/CM1.mat;

ho = 0.2;
opts.k = 5;
opts.N  = 10;
opts.T  = 100;

% run 10 times for average result
metrics = zeros(10, 6);
% for i = 1:10
HO = cvpartition(label,'HoldOut',ho); 
opts.Model = HO;

% feature selection method
alg = 'eo';

file = 'CM1'; 

FS     = jfs(alg, feat, label, opts);

% load AEEEM/KNN/abo/abo_EQ.mat

sf_idx = FS.sf;


% % KNN 
rs_knn = jknn(feat(:, sf_idx),label,opts);
fprintf('\n --------- Evaluating KNN --------- \n');
metrics_knn = evaluate(rs_knn);
saved('KNN', alg, file, metrics_knn, sf_idx);
% 
% DT 
% rs_dt = dt(feat(:, sf_idx),label,opts);
% fprintf('\n --------- Evaluating DT --------- \n');
% metrics_dt = evaluate(rs_dt);
% saved('DT', alg, file, metrics_dt, sf_idx);


% % LR 
% rs_lr = lr(feat(:, sf_idx), label, opts);
% fprintf('\n --------- Evaluating LR --------- \n');
% metrics_lr = evaluate(rs_lr);
% saved('LR', alg, file, metrics_lr, sf_idx);
% 
% % RF
% rs_rf = rf(feat(:, sf_idx), label, opts);
% fprintf('\n --------- Evaluating RF --------- \n');
% metrics_rf = evaluate(rs_rf);
% saved('RF', alg, file, metrics_rf, sf_idx);
% 
% % ET 
% rs_et = extratree(feat(:, sf_idx), label, opts);
% fprintf('\n --------- Evaluating ET --------- \n');
% metrics_et = evaluate(rs_et);
% saved('ET', alg, file, metrics_et, sf_idx);
% 
% % AB 
% rs_ab = adaboost(feat(:, sf_idx), label, opts);
% fprintf('\n --------- Evaluating AB --------- \n');
% metrics_ab = evaluate(rs_ab);
% saved('AB', alg, file, metrics_ab, sf_idx);

% final_metrics = metrics_rf;
% 
% metrics(i, 1) = final_metrics.acc;
% metrics(i, 2) = final_metrics.precision;
% metrics(i, 3) = final_metrics.recall;
% metrics(i, 4) = final_metrics.f1;
% metrics(i, 5) = final_metrics.auc;
% metrics(i, 6) = sf_idx;
% % 
% % end
% % 
% avg_results = mean(metrics, 1);  % Compute mean along columns
% 
% disp('Average results according to columns:');
% disp(avg_results);
% % 
% 
% 
% acc = avg_results(1);
% precision = avg_results(2);
% recall = avg_results(3);
% f1 = avg_results(4);
% auc = avg_results(5);






% save AEEEM/RF/baseline/PDE.mat acc precision recall f1 auc sf_idx


% fprintf('\n Number of selected features: %d / %d \n', length(sf_idx), size(feat, 2));




