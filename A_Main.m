% Wrapper Feature Selection Toolbox 
 
% There are more than 40 wrapper FS methods are offered 
% You may open < List_Method.m file > to check all available methods

%---Usage-------------------------------------------------------------
% If you wish to use 'PSO' (see example 1) then you write
% FS = jfs('pso',feat,label,opts);

% If you want to use 'SMA' (see example 2) then you write
% FS = jfs('sma',feat,label,opts);

% * All methods have different calling name (refer List_Method.m file)


%---Input-------------------------------------------------------------
% feat   : Feature vector matrix (Instances x Features)
% label  : Label matrix (Instances x 1)
% opts   : Parameter settings 
% opts.N : Number of solutions / population size (* for all methods)
% opts.T : Maximum number of iterations (* for all methods)
% opts.k : Number of k in k-nearest neighbor 

% Some methods have their specific parameters (example: PSO, GA, DE) 
% if you do not set them then they will define as default settings
% * you may open the < m.file > to view or change the parameters
% * you may use 'opts' to set the parameters of method (see example 1)
% * you may also change the < jFitnessFunction.m file >


%---Output------------------------------------------------------------
% FS    : Feature selection model (It contains several results)
% FS.sf : Index of selected features
% FS.ff : Selected features
% FS.nf : Number of selected features
% FS.c  : Convergence curve
% Acc   : Accuracy of validation model


%% Example 1: Particle Swarm Optimization (PSO) 
clear, clc, close;
% Number of k in K-nearest neighbor
opts.k = 6;
% Ratio of validation data
ho = 0.2;
% Common parameter settings 
opts.N  = 10;     % number of solutions
opts.T  = 100;    % maximum number of iterations
% Parameters of PSO
opts.c1 = 2;
opts.c2 = 2;
opts.w  = 0.9;
% Load dataset
load AEEEM/eq.mat; 
% Divide data into training and validation sets
HO = cvpartition(label,'HoldOut',ho); 
opts.Model = HO; 
% Perform feature selection 
FS     = jfs('pso',feat,label,opts);
% Define index of selected features
sf_idx = FS.sf;
% metrics  
prediction    = jknn(feat(:,sf_idx),label,opts); 
acc = sum(prediction.pred == prediction.label) / length(prediction.label);
pred = prediction.pred.';
label = prediction.label.';
pred_prob = prediction.pred_prob;

fprintf('\n Accuracy: %g %%', 100 * acc);
% Plot convergence
plot(FS.c); grid on;
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('PSO');

% save selected data
save AEEEM/PSO/pso_EQ.mat sf_idx pred label pred_prob


%% Example 2: Slime Mould Algorithm (SMA) 
clear, clc, close;
% Number of k in K-nearest neighbor
opts.k = 5; 
% Ratio of validation data
ho = 0.2;
% Common parameter settings 
opts.N  = 10;     % number of solutions
opts.T  = 100;    % maximum number of iterations
% Load dataset
load ionosphere.mat; 
% Divide data into training and validation sets
HO = cvpartition(label,'HoldOut',ho); 
opts.Model = HO; 
% Perform feature selection 
FS     = jfs('sma',feat,label,opts);
% Define index of selected features
sf_idx = FS.sf;
% Accuracy  
Acc    = jknn(feat(:,sf_idx),label,opts); 
% Plot convergence
plot(FS.c); grid on;
xlabel('Number of Iterations');
ylabel('Fitness Value'); 
title('SMA');


%% Example 3: Whale Optimization Algorithm (WOA) 
clear, clc, close;
% Number of k in K-nearest neighbor
opts.k = 5; 
% Ratio of validation data
ho = 0.2;
% Common parameter settings 
opts.N = 10;     % number of solutions
opts.T = 100;    % maximum number of iterations
% Parameter of WOA
opts.b = 1;
% Load dataset
load ionosphere.mat; 
% Divide data into training and validation sets
HO = cvpartition(label,'HoldOut',ho); 
opts.Model = HO; 
% Perform feature selection 
FS     = jfs('woa',feat,label,opts);
% Define index of selected features
sf_idx = FS.sf;
% Accuracy  
Acc    = jknn(feat(:,sf_idx),label,opts); 
% Plot convergence
plot(FS.c); grid on;
xlabel('Number of Iterations'); 
ylabel('Fitness Value'); 
title('WOA');

%% Equilibrium Optimizer (EO)
clear, clc, close
% Number of k in K-nearest neighbor
opts.k = 5;
% Ratio of validation data
ho = 0.2;
% Common parameter settings 
opts.N  = 10;     % number of solutions
opts.T  = 100;    % maximum number of iterations
% Parameters of EO

% Load dataset
load AEEEM/mat/PDE.mat; 
% Divide data into training and validation sets
HO = cvpartition(label,'HoldOut',ho); 
opts.Model = HO; 
% Perform feature selection 
FS     = jfs('eo',feat,label,opts);
% Define index of selected features
sf_idx = FS.sf;
% metrics  
prediction    = jknn(feat(:,sf_idx),label,opts); 
acc = sum(prediction.pred == prediction.label) / length(prediction.label);
pred = prediction.pred.';
label = prediction.label.';
pred_prob = prediction.pred_prob;

fprintf('\n Accuracy: %g %%', 100 * acc);
% Plot convergence
plot(FS.c); grid on;
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('PSO');

% save selected data
if isfile("AEEEM/eo/eo_PDE.mat")
    save AEEEM/eo/eo_PDE_1.mat sf_idx pred label pred_prob;
else 
    save AEEEM/eo/eo_PDE.mat sf_idx pred label pred_prob;
end

%% Atom Search Optimization (ASO)
clear, clc, close
% Number of k in K-nearest neighbor
opts.k = 5;
% Ratio of validation data
ho = 0.2;
% Common parameter settings 
opts.N  = 10;     % number of solutions
opts.T  = 100;    % maximum number of iterations
% Parameters of EO

% Load dataset
load AEEEM/modified/PDE.mat; 
% Divide data into training and validation sets
HO = cvpartition(label,'HoldOut',ho); 
opts.Model = HO; 
% Perform feature selection 
FS     = jfs('aso',feat,label,opts);
% Define index of selected features
sf_idx = FS.sf;
% metrics  
prediction    = jknn(feat(:,sf_idx),label,opts); 
acc = sum(prediction.pred == prediction.label) / length(prediction.label);
pred = prediction.pred.';
label = prediction.label.';
pred_prob = prediction.pred_prob;

fprintf('\n Accuracy: %g %%', 100 * acc);
% Plot convergence
plot(FS.c); grid on;
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('ASO');

% save selected data
if isfile("AEEEM/KNN/aso/aso_PDE.mat")
    save AEEEM/KNN/aso/aso_PDE_1.mat sf_idx pred label pred_prob;
else 
    save AEEEM/KNN/aso/aso_PDE.mat sf_idx pred label pred_prob;
end

%% Henry Gas Solubility Optimization (HGSO)
clear, clc, close
% Number of k in K-nearest neighbor
opts.k = 5;
% Ratio of validation data
ho = 0.2;
% Common parameter settings 
opts.N  = 10;     % number of solutions
opts.T  = 100;    % maximum number of iterations
% Parameters of EO

% Load dataset
load AEEEM/mat/PDE.mat; 
% Divide data into training and validation sets
HO = cvpartition(label,'HoldOut',ho); 
opts.Model = HO; 
% Perform feature selection 
FS     = jfs('hgso',feat,label,opts);
% Define index of selected features
sf_idx = FS.sf;
% metrics  
prediction    = jknn(feat(:,sf_idx),label,opts); 
acc = sum(prediction.pred == prediction.label) / length(prediction.label);
pred = prediction.pred.';
label = prediction.label.';
pred_prob = prediction.pred_prob;

fprintf('\n Accuracy: %g %%', 100 * acc);
% Plot convergence
plot(FS.c); grid on;
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('ASO');

% save selected data
if isfile("AEEEM/hgso/hgso_PDE.mat")
    save AEEEM/hgso/hgso_PDE_1.mat sf_idx pred label pred_prob;
else 
    save AEEEM/hgso/hgso_PDE.mat sf_idx pred label pred_prob;
end

%% Artificial Butterfly Optimization (ABO)
clear, clc, close
% Number of k in K-nearest neighbor
opts.k = 5;
% Ratio of validation data
ho = 0.2;
% Common parameter settings 
opts.N  = 10;     % number of solutions
opts.T  = 100;    % maximum number of iterations
% Parameters of ABO

% Load dataset
load AEEEM/mat/PDE.mat; 
% Divide data into training and validation sets
HO = cvpartition(label,'HoldOut',ho); 
opts.Model = HO; 
% Perform feature selection 
FS     = jfs('abo',feat,label,opts);
% Define index of selected features
sf_idx = FS.sf;
% metrics  
prediction    = jknn(feat(:,sf_idx),label,opts); 
acc = sum(prediction.pred == prediction.label) / length(prediction.label);
pred = prediction.pred.';
label = prediction.label.';
pred_prob = prediction.pred_prob;

fprintf('\n Accuracy: %g %%', 100 * acc);
% Plot convergence
plot(FS.c); grid on;
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('ASO');

% save selected data
if isfile("AEEEM/abo/abo_PDE.mat")
    save AEEEM/abo/abo_PDE_1.mat sf_idx pred label pred_prob;
else 
    save AEEEM/abo/abo_PDE.mat sf_idx pred label pred_prob;
end

%% Poor And Rich Optimization (PRO)
clear, clc, close
% Number of k in K-nearest neighbor
opts.k = 5;
% Ratio of validation data
ho = 0.2;
% Common parameter settings 
opts.N  = 10;     % number of solutions
opts.T  = 100;    % maximum number of iterations
% Parameters of MPA
opts.P     = 0.5;    % constant
opts.FADs  = 0.2;

% Load dataset
load AEEEM/modified/EQ.mat; 
% Divide data into training and validation sets
HO = cvpartition(label,'HoldOut',ho); 
opts.Model = HO; 
% Perform feature selection 
FS     = jfs('mpa',feat,label,opts);
% Define index of selected features
sf_idx = FS.sf;
% metrics  
prediction    = jknn(feat(:,sf_idx),label,opts); 
acc = sum(prediction.pred == prediction.label) / length(prediction.label);
pred = prediction.pred.';
label = prediction.label.';
pred_prob = prediction.pred_prob;
evaluate(prediction)
fprintf('\n Accuracy: %g %%', 100 * acc);
% Plot convergence
plot(FS.c); grid on;
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('ASO');


% save selected data
if isfile("AEEEM/KNN/mpa/EQ.mat")
    save AEEEM/KNN/mpa/EQ_1.mat sf_idx pred label pred_prob;
else 
    save AEEEM/KNN/mpa/EQ.mat sf_idx pred label pred_prob;
end




