function [Sigma_structural,p] = FUN_learning_cov_est(C,M,p,update_mode)
% Description:    This function estimates the covariance matrix of the data by imposing
%                 different kind of structures for spatial or temporal correlation matrices. 
	 
% Input

% Output:
% 	Sigma_structural: Final estimated covariance


switch update_mode
    case 'Geodesic'
%% -----------------------------
%       update_mode = 'Geodesic'; 
%       S = inv(sqrtm(C))*sqrtm((sqrtm(C))*M*(sqrtm(C)))*inv(sqrtm(C));

        % Efficient Implementation 
        eps_default = 1e-5; 
        [b_vec,b_val] = eig(C);
        root_C_coeff = sqrt(max(real(diag(b_val)),0));

        inv_root_C_coeff = zeros(size(C,1),1);
        inv_root_C_index = find(root_C_coeff >= eps_default);
        inv_root_C_coeff(inv_root_C_index) = 1./root_C_coeff(inv_root_C_index);

        root_C = b_vec * diag(root_C_coeff) * b_vec';
        inv_root_C = b_vec*diag(inv_root_C_coeff)*b_vec';

        [a_vec,a_val] = eig(root_C * M * root_C);
        A_coeff = sqrt(max(real(diag(a_val)),0));
        A = a_vec * diag(A_coeff) * a_vec';
        S = inv_root_C * A * inv_root_C;  

		%% ---------------------------------------------------------------------
		%         % Efficient and Geodesic Implementation compatible with the paper
		%         % Please change the C_M and C_N in the main code. Use their
		%         % inverse, e.g., C_source = inv(L' * SigmaY_inv * L) and 
		%         % C_noise = SigmaY_estimated. 
				
		% 		  % Geodesic
		%         S = sqrtm(C)*sqrtm(inv(sqrtm(C))*M*inv(sqrtm(C)))*sqrtm(C);
		
				  % Efficient
		%         eps_default = 1e-8; 
		%         [b_vec,b_val] = eig(C);
		%         root_C_coeff = sqrt(max(real(diag(b_val)),0));
		% 
		%         inv_root_C_coeff = zeros(size(C,1),1);
		%         inv_root_C_index = find(root_C_coeff >= eps_default);
		%         inv_root_C_coeff(inv_root_C_index) = 1./root_C_coeff(inv_root_C_index);
		% 
		%         root_C = b_vec * diag(root_C_coeff) * b_vec';
		%         inv_root_C = b_vec*diag(inv_root_C_coeff)*b_vec';
		% 
		%         [a_vec,a_val] = eig(inv_root_C * M * inv_root_C);
		%         A_coeff = sqrt(max(real(diag(a_val)),0));
		%         A = a_vec * diag(A_coeff) * a_vec';
		%         S = root_C * A * root_C;
		%% ---------------------------------------------------------------------       

    case 'Diagonal'
        % solving inner problem using diagonal constraint
        h = diag(C); 
        g = diag(M);    
        p = sqrt(g./(h));
        S = diag(p);
		
end 
%%
Sigma_structural = S; 

end