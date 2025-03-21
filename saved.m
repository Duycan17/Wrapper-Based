%% SAVE METRICS 
function saved(clsf, alg, file, metrics, sf_idx)

acc = metrics.acc;
precision = metrics.precision;
recall = metrics.recall;
f1 = metrics.f1;
auc = metrics.auc;

dirPath = fullfile('AEEEM', clsf, alg);
filePath = fullfile(dirPath, [file, '.mat']);
if ~exist(dirPath, 'dir')
    mkdir(dirPath);
end
    
save(filePath, 'acc', 'precision', 'recall', 'f1', 'auc', 'sf_idx');

end