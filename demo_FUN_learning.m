clc
clear
close all

num_trials = 1; 
% Uncomment below for having the statistical analysis results.  
% num_trials = 100; 

for trial = 1:num_trials
 
    %% ---------------------------------------------
    % Setting directoris. 
    current_dir = pwd;
    cd(current_dir);
    addpath(genpath([current_dir '/utils/']))

    %% ---------------------------------------------
    % Setting the ground truth (GT) parameters

    N = 2004;   % Number of voxels in source space (when using the 1D leadfiled) 
    T = 200;    % Number of time samples
    M = 58;     % Number of sensors

    stdnoise = 0.1; % The power of noise and source will be rescaled later according to the respective SNR.

    % ---------------------------------------------
    % Generate the forward operator. 
    % Lead filed matrix
    load([current_dir '\data\data1D.mat'], 'L', 'D', 'loc')

    % Random Gaussian matrix
    % L = randn(M,N); 

    % load head model and some miscallaneous data
    load('data/sa')
    load('data/cm17')

    %% ---------------------------------------------
    % Choose one of the following settings and chagne the updating rules, accordingly. 

    % algo_name = 'diag_groundtruth_diag_estimation';
    % algo_name = 'full_groundtruth_diag_estimation';

    % algo_name = 'diag_groundtruth_geodesic_estimation';
    algo_name = 'full_groundtruth_geodesic_estimation';

    %% ---------------------------------------------
    %  ---- Covarinace matrix of source ------------ 

    % Diagonal and Sparse 
    [Gamma_0,indice] = cov_mat_gen(N,'sparse', 10); 

    %  ----  Covarinace matrix of noise ------------  
    % Full-structural 
    [Lambda_root, Lambda_0]  = randpsd(M); 

    % Diagonal
    % Lambda_0 = cov_mat_gen(M,'diagonal', stdnoise^2);

    %% ---------------------------------------------
    % Generating the covariance matrices and time-series in source and sensor space

    SigmaY = Lambda_0 + (L * Gamma_0 * L');
    % rng(1);
    % rng('default');
    X_0 = mvnrnd(zeros(N,1),Gamma_0,T)'; 
    Y = L * X_0; 
    norm_signal = norm(Y, 'fro');
    noise = mvnrnd(zeros(M,1),Lambda_0,T)'; 

    norm_noise = norm(noise, 'fro'); 
    noise = noise ./ norm_noise; 

    % SNR (dB) based on the energy ration between signal and noise
    alpha = 0.5; 
    SNR_value = 20*log10(alpha/(1-alpha));
    Y_total = Y + (1-alpha)*noise*norm_signal/alpha;
    EEG_baseline = (1-alpha)*noise*norm_signal/alpha;

    scale = (1-alpha)*norm_signal/(alpha*norm_noise); 
    Lambda_0 = (scale^2) * Lambda_0; 

    %% ---------------------------------------------
    % Normalizing the leadfiled
    DW = sqrt(sum(L.^2));
    L = L*diag(1./DW);
    N0 = 5;
    X_true = X_0; 
    plot_figures = true; 

    %% ----------------------------------------------
    % Plotting the source and noise covariances as well as voxel distributions
    if plot_figures
        f1 = figure('Name','Voxles', 'units','normalized','innerposition',[0.5 0 0.45 0.45]);
%       suptitle('\fontsize{20}Voxel Distributions')
        subplot(2,1,1);
        amp_origin = sqrt(sum(X_0.^2, 2));
        amp_origin = amp_origin ./ sum(amp_origin);
        plot((1:N),amp_origin);
        xlabel('voxel index');
        set(gca(),'XLim',[1 N]);
        % set(gca(),'YLim',[0 inf]);
        title('\fontsize{16}Ground truth') 
        drawnow

%       figure(2)
        f2 = figure('Name','Noise Covariance', 'units','normalized','innerposition',[0 0.45 0.45 0.45]);
        subplot(1,2,1);
%       suptitle('Noise Covariance')
        imagesc(real(Lambda_0))
        title('\fontsize{16}Ground truth') 
        axis('equal')
        axis('tight')
        colorbar
        drawnow

%         figure(3)
%         figure('units','normalized','innerposition',[0.5 0 0.5 0.5])
%         subplot(1,2,1);
%         imagesc(real(Gamma_0))
%         title('\fontsize{16}Ground truth') 
%         axis('equal')
%         axis('tight')
%         colorbar
%         drawnow
    end 

    %% =============================================== %% 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%% =========== FUN Learning ==================== %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    noise_cov = cov(EEG_baseline');
    tic
    [Weight_FUN,Gamma_FUN,NegLogLikelihood_FUN,Lambda_FUN,W_FUN]= ...
        FUN_learning_func(L,Y_total,'noise_cov',noise_cov,'Lambda_0',Lambda_0,'Gamma_0',Gamma_0,'noise_update_mode','Geodesic',...
        'source_update_mode','Diagonal','max_num_iterations',300,'threshold_error',1e-7,'print_results',1,'print_figures',1);
    time_FUN = toc; 

    % Weight_FUN = real(Weight_FUN); 
    Weight_FUN = diag(1./DW) * Weight_FUN;  

    corr_Lambda_FUN = corr(real(Lambda_FUN(:)),Lambda_0(:));
    NMSE_Lambda_FUN = norm(Lambda_0-real(Lambda_FUN),'fro')^2/norm(Lambda_0,'fro')^2;

    [F1measure_FUN] = calc_F1measure(Weight_FUN,X_true);

    MSE_FUN = (norm(Weight_FUN-X_true,'fro')/norm(X_true,'fro'))^2;
    [EM_FUN,MEAN_CO_FUN,MEAN_CO_DIST_FUN] = perf_emd(Weight_FUN, X_true, D, indice);

    fprintf('FUN Learning: MSE = %3.2f; Lambda-Corr = %4.3f; F1 = %4.3f; EMD = %4.3f; CORR = %4.3f; EUCL = %4.3f; Time = %4.3f \n',...
       MSE_FUN, corr_Lambda_FUN, F1measure_FUN,  EM_FUN,  MEAN_CO_FUN, MEAN_CO_DIST_FUN, time_FUN);

    if plot_figures
        method = 'FUN-Learning'; 
        Weight = real(Weight_FUN); 
        noise_cov = real(Lambda_FUN); 
        figure_gen_noise_cov_brain_fig(current_dir, algo_name, method, sa, cm17a, Weight,noise_cov)
    end 

    %% =============================================== %% 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%% ========= Heteroscedastic Champagne ========= %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
    noise_cov = cov(EEG_baseline');
    tic
    [Weight_Champ_Heterosc,Gamma_Champ_Heterosc,NegLogLikelihood_Champ_Heterosc,Lambda_Champ_Heterosc,W_Champ_Heterosc]= ...
        FUN_learning_func(L,Y_total,'noise_cov',noise_cov,'Lambda_0',Lambda_0,'Gamma_0',Gamma_0,'noise_update_mode','Diagonal',...
        'source_update_mode','Diagonal','max_num_iterations',300,'threshold_error',1e-7,'print_results',0,'print_figures',0);
    time_Champ_Heterosc = toc; 

    % Weight_Champ_Heterosc = real(Weight_Champ_Heterosc); 
    Weight_Champ_Heterosc = diag(1./DW) * Weight_Champ_Heterosc;  

    corr_Lambda_Champ_Heterosc = corr(real(Lambda_Champ_Heterosc(:)),Lambda_0(:));
    NMSE_Lambda_Champ_Heterosc = norm(Lambda_0-real(Lambda_Champ_Heterosc),'fro')^2/norm(Lambda_0,'fro')^2;

    [F1measure_Champ_Heterosc] = calc_F1measure(Weight_Champ_Heterosc,X_true);

    MSE_Champ_Heterosc = (norm(Weight_Champ_Heterosc-X_true,'fro')/norm(X_true,'fro'))^2;
    [EM_Champ_Heterosc,MEAN_CO_Champ_Heterosc,MEAN_CO_DIST_Champ_Heterosc] = perf_emd(Weight_Champ_Heterosc, X_true, D, indice);

    fprintf('Heteroscedastic Champagne: MSE = %3.2f; Lambda-Corr = %4.3f; F1 = %4.3f; EMD = %4.3f; CORR = %4.3f; EUCL = %4.3f; Time = %4.3f \n',...
       MSE_Champ_Heterosc, corr_Lambda_Champ_Heterosc, F1measure_Champ_Heterosc,  EM_Champ_Heterosc,  MEAN_CO_Champ_Heterosc, MEAN_CO_DIST_Champ_Heterosc, time_Champ_Heterosc);

    if plot_figures
        method = 'Champ-Heterosc'; 
        Weight = real(Weight_Champ_Heterosc); 
        noise_cov = real(Lambda_Champ_Heterosc); 
        figure_gen_noise_cov_brain_fig(current_dir, algo_name, method, sa, cm17a, Weight,noise_cov)
    end 


    %% =============================================== %% 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%% ========= Homoscedastic Champagne =========== %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
    noise_cov = cov(EEG_baseline');
    tic
    [Weight_Champ_Homosc,Gamma_Champ_Homosc,NegLogLikelihood_Champ_Homosc,Lambda_Champ_Homosc,W_Champ_Homosc]= ...
        Champ_Homoscedastic(L,Y_total,'noise_cov',noise_cov,'Lambda_0',Lambda_0,'Gamma_0',Gamma_0,...
        'max_num_iterations',300,'threshold_error',1e-7,'print_results',0,'print_figures',0);
    time_Champ_Homosc = toc; 

    % Weight_Champ_Homosc = real(Weight_Champ_Homosc); 
    Weight_Champ_Homosc = diag(1./DW) * Weight_Champ_Homosc;  

    corr_Lambda_Champ_Homosc = corr(real(Lambda_Champ_Homosc(:)),Lambda_0(:));
    NMSE_Lambda_Champ_Homosc = norm(Lambda_0-real(Lambda_Champ_Homosc),'fro')^2/norm(Lambda_0,'fro')^2;

    [F1measure_Champ_Homosc] = calc_F1measure(Weight_Champ_Homosc,X_true);

    MSE_Champ_Homosc = (norm(Weight_Champ_Homosc-X_true,'fro')/norm(X_true,'fro'))^2;
    [EM_Champ_Homosc,MEAN_CO_Champ_Homosc,MEAN_CO_DIST_Champ_Homosc] = perf_emd(Weight_Champ_Homosc, X_true, D, indice);

    fprintf('Homoscedastic Champagne: MSE = %3.2f; Lambda-Corr = %4.3f; F1 = %4.3f; EMD = %4.3f; CORR = %4.3f; EUCL = %4.3f; Time = %4.3f \n',...
       MSE_Champ_Homosc, corr_Lambda_Champ_Homosc, F1measure_Champ_Homosc,  EM_Champ_Homosc,  MEAN_CO_Champ_Homosc, MEAN_CO_DIST_Champ_Homosc, time_Champ_Homosc);

    if plot_figures
        method = 'Champ-Homosc'; 
        Weight = real(Weight_Champ_Homosc); 
        noise_cov = real(Lambda_Champ_Homosc); 
        figure_gen_noise_cov_brain_fig(current_dir, algo_name, method, sa, cm17a, Weight,noise_cov)
    end 

end

%% -------------------------------------------------- 
% Plot the convergence 
 
figure('units','normalized','outerposition',[0 0 0.65 0.85])
h_homosc = plot(1:length(NegLogLikelihood_Champ_Homosc),NegLogLikelihood_Champ_Homosc);
set(h_homosc, 'LineStyle', '--', 'LineWidth', 8 , 'Color', 'r');

hold on
h_heterosc = plot(1:length(NegLogLikelihood_Champ_Heterosc),NegLogLikelihood_Champ_Heterosc);
set(h_heterosc,'LineStyle', '-', 'LineWidth', 8, 'Color', 'green');

hold on
h_full = plot(1:length(NegLogLikelihood_FUN),NegLogLikelihood_FUN);
set(h_full, 'LineStyle', ':', 'LineWidth', 8, 'Color', 'b');
  
hLegend = legend( ...
[h_homosc, ...  
h_heterosc, ... 
h_full], ...
'Homoscedastic', ...
'Heteroscedastic' , ...
'Full Structure'); 
 
set(hLegend,'interpreter','tex');
set(hLegend,'Location','Northeast');
set([hLegend, gca]             , ...
    'FontSize'   , 40           );

set(gca, ...
  'Box'         , 'off'     , ...
  'TickDir'     , 'in'     , ...
  'YMinorTick'  , 'on'      , ...
  'TickLength'  , [.01 .05] , ...
  'YGrid'       , 'on'      , ...
  'XMinorTick'  , 'on'      , ...
  'YMinorGrid'  , 'on'      , ...
  'XGrid'       , 'on'      , ...
  'XMinorTick'  , 'on'      , ...
  'TickLength'  , [.01 .05] , ...
  'FontSize'    , 60        , ... 
  'LineWidth'   , 1         ); 
set(gca, 'XScale', 'log')
set(gcf,'color','w');

hXLabel = xlabel('Number of iterations');
hYLabel = ylabel('Loss');
set(hXLabel,'Interpreter','tex');
set(hYLabel,'Interpreter','tex');
set(hXLabel,'FontSize',60);
set(hYLabel,'FontSize',60);

res = 100; 
file_name = [current_dir '/figures/' algo_name '/Negloglikelihood_loss']; 
savefig(file_name); 
export_fig(file_name, ['-r' num2str(res)], '-a2', '-transparent', '-eps', '-pdf'); 
res = 150; 
export_fig(file_name,['-r' num2str(res)], '-png'); 

