clear, clc, close

% List of all datasets to process
% datasets = {'CM1','EQ','JDT','KC1','KC2','Lucene','Mylyn', 'PC1', 'PDE'};
datasets = {'PDE'};
% List of algorithms and classifiers
algorithms = {'gndo','sma', 'hho', 'pfa'};
% algorithms = {'mrfo'}
% classifiers = {'RF', 'LR', 'KNN', 'ET', 'AB','DT'}; % RF = Random Forest, LR = Logistic Regression, ET = Extra Trees, AB = AdaBoost
classifiers = {'LR'};
% Common parameter settings
opts.k = 5; % for KNN
opts.N = 10; % number of solutions
opts.T = 100; % maximum number of iterations
ho = 0.2; % holdout ratio
opts.S  = 2;     % somersault factor 

% Initialize total null count
total_null_count = 0;

% Main loop for all combinations
for i = 1:length(datasets)
    file = datasets{i};
    fprintf('\nProcessing dataset: %s\n', file);
    
    try
        % Load dataset
        load(['AEEEM/modified/' file '.mat']);
        
        % Create holdout partition
        HO = cvpartition(label, 'HoldOut', ho);
        opts.Model = HO;
        
        % Loop through algorithms
        for alg_idx = 1:length(algorithms)
            alg = algorithms{alg_idx};
            fprintf('\nRunning algorithm: %s\n', upper(alg));
            
            % Perform feature selection
            FS = jfs(alg, feat, label, opts);
            sf_idx = FS.sf;
            
            % Loop through classifiers
            for clf_idx = 1:length(classifiers)
                clf = classifiers{clf_idx};
                fprintf('Classifier: %s\n', clf);
                
                % Select appropriate classifier function
                switch clf
                    case 'RF'
                        prediction = rf(feat(:,sf_idx), label, opts);
                    case 'LR'
                        prediction = lr(feat(:,sf_idx), label, opts);
                    case 'KNN'
                        prediction = jknn(feat(:,sf_idx), label, opts);
                    case 'ET'
                        prediction = extratree(feat(:,sf_idx), label, opts);
                    case 'AB'
                        prediction = adaboost(feat(:,sf_idx), label, opts);
                    case 'DT'
                        prediction = dt(feat(:,sf_idx), label, opts);
                end
                
                % Calculate metrics
                acc = sum(prediction.pred == prediction.label) / length(prediction.label);
                pred = prediction.pred.';
                current_label = prediction.label.';
                pred_prob = prediction.pred_prob;
                metrics = evaluate(prediction);
                
                % Count null metrics
                metric_fields = fieldnames(metrics);
                null_count = 0;
                for field_idx = 1:length(metric_fields)
                    if isnan(metrics.(metric_fields{field_idx}))
                        null_count = null_count + 1;
                        fprintf('Null metric found in %s: %s\n', metric_fields{field_idx}, file);
                    end
                end
                fprintf('Number of null metrics: %d\n', null_count);
                total_null_count = total_null_count + null_count;
                
                fprintf('Accuracy: %g %%\n', 100 * acc);
                
                % Plot convergence
                % figure('Name', sprintf('Convergence Plot - %s - %s - %s', file, upper(alg), clf));
                % plot(FS.c); grid on;
                % xlabel('Number of Iterations');
                % ylabel('Fitness Value');
                % title(sprintf('%s - %s - %s', file, upper(alg), clf));
                
                % Save results
                saved(clf, alg, file, metrics, sf_idx);
            end
        end
        
    catch ME
        fprintf('Error processing combination - Dataset: %s\nError: %s\n', file, ME.message);
        continue;
    end
end

fprintf('\nAnalysis completed for all combinations!\n');
fprintf('\nTotal number of null metrics across all combinations: %d\n', total_null_count);