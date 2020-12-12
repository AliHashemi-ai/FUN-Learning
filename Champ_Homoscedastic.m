function [X,Gamma,NegLogLikelihood,Lambda,W] = Champ_Homoscedastic(L, Y, varargin)
% Full-structural Noise (FUN) Learning Algorithm. 

% ===== INPUTS =====
%   L            			: M X N Lead field matrix
%
%   Y            			: M X T measurement matrix
%
%   noise_cov     			: M X M Initial value for the noise covariance matrix. 
%
%   Lambda_0    			: M X M Ground-truth noise covariance matrix. 
%
%   Gamma_0     			: N X N Ground-truth source covariance matrix.

%  'max_num_iterations' 	: Maximum number of iterations.
%
%  'threshold_error'    	: Threshold to stop the whole algorithm. 
%
%  'print_results'      	: Display flag for showing the progress results. 
%
%  'print_figures'      	: Display flag for showing the plots of voxels, source and noise covariances. 
%
% ===== OUTPUTS =====
%   X                  : The estimated solution matrix, or called source matrix (size: N X T)
%   Gamma              : Estimated diagonal source covarinace matrix.
%   Lambda             : Estimated full-structured noise covarinace matrix.  
%   NegLogLikelihood   : The negative log-likelihood value (cost function).
%   W   			   : Filter for generating the final source resconstruction.

% --- Referecnes ---
%     [1] A.  Hashemi,  C.  Cai,  G.  Kutyniok,  K.-R.  Meuller,  S.  Nagarajan,  and S.  Haufe 
%     "Unification  of  sparse  Bayesian  learning  algorithms  for electromagnetic  brain  imaging  
%     with  the  majorization  minimizationframework,"
%     bioRxiv, 2020

%     [2] D.P. Wipf, J.P. Owen, H.T. Attias, K. Sekihara, and S. Nagarajan, 
%	  "Robust Bayesian Estimation of the Location, Orientation, and Time 
%	  course of Multiple Correlated Neural Sources using MEG,"
%     Processing, Vol.55, No.7, 2007.
 
 
% Dimension of the Problem
[M,N] = size(L); 
[~,T] = size(Y);  

% Default Control Parameters  
Threshold_error = 1e-8;         % threshold for stopping the algorithm. 
Max_num_iterations = 300;       % maximum number of iterations
print_results = 0;          	% don't show progress information
print_figures = 0; 				% plot the amplitude of the sources and their corresponding estimation
								% along with the plots for oroginal and estimated noise and source covariances

itr_count = 1; % number of iterations used

% Random initialization for Lambda as a default
tmp = rand(M,T);
Lambda_init = tmp*tmp';

YYt = Y * Y'; 
C_y = 1/T * YYt; % Sample covarinace matrix  
	
% Assigning the values according to the user input
if(mod(length(varargin),2)==1)
    error('Optional parameters should always go by pairs\n');
else
 
for i=1:2:(length(varargin)-1)
    switch lower(varargin{i})
		case 'noise_cov'
		Lambda_init = varargin{i+1}; 
        case 'gamma_0'
            Gamma_0 = varargin{i+1}; 
        case 'lambda_0'
            Lambda_0 = varargin{i+1}; 
        case 'threshold_error'   
            Threshold_error = varargin{i+1}; 
        case 'print_results'    
            print_results = varargin{i+1}; 
        case 'print_figures'    
            print_figures = varargin{i+1}; 
        case 'max_num_iterations'
            Max_num_iterations = varargin{i+1};  
        otherwise
            error(['Unrecognized parameter: ''' varargin{i} '''']);
    end
end

 
%% ===========================================
%% === Initialize Lambda and Gamma    ========
% ============================================

% Random initialization for Gamma
% tmp = rand(N,2*N);
% Gamma_init = tmp*tmp';

%% Pseudo-inverse initialization for Gamma
L_square=sum(L.^2,1);              
Inv_L_square=zeros(1,N);
Index_pos=find(L_square>0);
Inv_L_square(Index_pos)=1./L_square(Index_pos);
w_filter = spdiags(Inv_L_square',0,N,N)*L';
gamma_init = mean(mean((w_filter*Y).^2));
gammas = gamma_init * ones(N,1);
Gamma_init = spdiags(gammas,0,N,N);

Lambda = Lambda_init; 
Gamma = Gamma_init; 
  
X_old = 0;
print_convergence = false;
Converge_X = false;   

% *** Learning loop ***
while(~Converge_X)

	SigmaY_estimated = (L * Gamma * L') + (Lambda);
	SigmaY_inv = inv(SigmaY_estimated); 
	SigmaY_invL = SigmaY_inv * L;	
	X_estimated = Gamma * L' * SigmaY_inv * Y;   
	X_bar = X_estimated;    
 
    gammas_old = gammas;
	Gamma_old = Gamma;
    S = SigmaY_invL' * Y;
	gammas = gammas .* sqrt(mean(abs(S).^2,2) ./ sum(L .* SigmaY_invL)');
	Gamma = spdiags(gammas,0,N,N);
	
	Lambda_old = Lambda; 
 
    E_total = Y - L * X_bar;
    M_noise = sum(mean(abs(E_total).^2,2));
    Sigma_X = L' * inv(Lambda) * L + inv(Gamma_old); 
    C_noise = (M - N + sum(diag(inv(Sigma_X))./gammas_old)); 
    scale = M_noise / C_noise; 

    % --- for more efficient but less accurate version incomment below: ---  
    % Sigma_X_diag = gammas.*(1 - gammas.* sum(L .* SigmaY_invL)'); % The covariance of the posterior distribution.
    % scale = (norm(Y - L * X_bar,'fro')^2/ T) /(M - N + sum(Sigma_X_diag./gammas_old)); 

    Lambda = scale * eye(M); 
 
    if print_convergence 
        fprintf('Gamma iteration error: %d \n \n ',norm(Gamma-Gamma_old,'fro'));
    end
 
    if print_convergence 
        fprintf('Lambda itration error: %d \n \n ',norm(Lambda-Lambda_old,'fro'));
    end

    % ================ Calculate the Log-Likelihood ==================	
	logdet_SigmaY = logdet(SigmaY_estimated);
	assert(isreal(logdet_SigmaY),'cov(Y) is not positive !');
    NegLogLikelihood_value = (logdet_SigmaY) + abs(trace(C_y*SigmaY_inv)); % log likelihood of the Gaussian density
    NegLogLikelihood(itr_count) = NegLogLikelihood_value;
    fprintf('Iteration: %d  \n',itr_count); 

    Converge_X = norm(X_old-X_bar,'fro') < Threshold_error;
    dX = max(max(abs(X_old - X_bar)));   
    error_X = norm(X_bar-X_old,'fro')^2/norm(X_old,'fro')^2;
            
    if (print_results==1) 
        fprintf('Norm X change: %d \n',norm(X_old-X_bar,'fro'));
        fprintf('X change: %d \n \n',dX);
    end
    
    if dX < Threshold_error || (itr_count > Max_num_iterations)
        Converge_X = true; 
        X_itr_est_error(itr_count) = error_X; 
    end
	
	X_old = X_bar; 
	
    if (print_results==1) 
		NMSE_Lambda = norm(Lambda_0-real(Lambda),'fro')^2/norm(Lambda_0,'fro')^2; 
		NMSE_Gamma = norm(Gamma_0-real(Gamma),'fro')^2/norm(Gamma_0,'fro')^2;
    
		corr_Lambda = corr(real(Lambda(:)),Lambda_0(:));
		corr_Gamma = corr(real(Gamma(:)),Gamma_0(:)); 
		
		fprintf('Lambda Corr error from GT: %d \n ',corr_Lambda);
		fprintf('Lambda NMSE error fron GT: %d \n \n',NMSE_Lambda);
						
		fprintf('Gamma Corr error from GT: %d \n ',full(corr_Gamma));
		fprintf('Gamma NMSE error from GT: %d \n \n ',NMSE_Gamma);
    end

    
	if print_figures 
	    figure(1)
		subplot(2,1,2);
		amp_est = sqrt(sum(real(X_bar).^2, 2));
		amp_est = amp_est ./ sum(amp_est);
		plot((1:N),amp_est,'r');
		xlabel('voxel index');
		set(gca(),'XLim',[1 N]);
		title('\fontsize{16}Estimation') 
		drawnow
 
		
		figure(2)
		subplot(1,2,2);
		imagesc(real(Lambda))
		title('\fontsize{16}Estimation') 
		axis('equal')
		axis('tight')
		colorbar
		drawnow
 
        
%       figure(3)
% 		subplot(1,2,2);
% 		imagesc(real(Gamma))
% 		title('\fontsize{16}Estimation') 
% 		axis('equal')
% 		axis('tight')
% 		colorbar
% 		drawnow
        
    end

    if print_figures 
        figure(3)
        plot((1:itr_count),NegLogLikelihood(1:itr_count));
        title(['\fontsize{16}Neg-LogLikelihood: ' int2str(itr_count) ' / ' int2str(Max_num_iterations)]);
        xlabel('\fontsize{16}iteration');
        set(gca(),'XLim',[0 itr_count]);
        drawnow
    end
    
    itr_count = itr_count + 1; 
        
	end
end

SigmaY_estimated = (L * Gamma * L') + (Lambda);
W_filter = Gamma * L' * inv(SigmaY_estimated);

X = zeros(N,T);
W = zeros(N,M); 

X(1:N,:) = X_bar; 
W(1:N,:) = W_filter;

if (print_results) 
    fprintf('\nFinish running ...\n'); 
end

return;

end

