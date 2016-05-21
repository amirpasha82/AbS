% Implementing analysis-by-synthesis (AbS) quantization for compressed
% sensing (CS) measurements as reported in the paper 'Analysis-by-synthesis Quantization 
% for Compressed Sensing Measurements' by Amirpasha Shirazinia, Saikat
% Chatterjee and Mikael Skoglund submitted to IEEE Trans. SP, 2013.

% Written by: Amirpasha Shirazinia, Communication Theory Lab, KTH
% Email: amishi@kth.se
% Created: May 18, 2013

clc
clear all
% close all
%% Setup
M = 100; % Size of vector x
meas_rate = 0.5; % Measurement rate
NN = round(M*meas_rate); % Number of measurements
MC = 10; % Number of Monte-Carlo realizations
K = 10; % Sparsity level
r_x = 1; % bit/scalar of x (r_x)
itr_max = 10; % Maximum number of iterations of the AbS method

%% Pre-allocations
MSE_near = zeros(MC,length(meas_rate));
MSE_abs_seq = zeros(MC,length(meas_rate));
% MSE_abs_nonseq = zeros(MC,length(meas_rate));
% MSE_abs_joint = zeros(MC,length(meas_rate));


% Measurement loop
for n=1:length(NN)
    N = NN(n)
    
    %% Codebook initialization
    r_y = floor(M*r_x/N); % Number of quantization bits for each measurement entry (r_y)
    % The following three steps are necessary for the case where 'b' is not an integer
    b_ex = floor(M*r_x - r_y*N); 
    b_1 = floor(M*r_x/N) + 1; 
    b_2 = floor(M*r_x/N);
    
    
        
    %% Read codebooks from offline-designed files (codepoints are design for a standard Gaussian scalar)
    C1 =  textread(strcat('cb_',num2str(2^b_1), '.txt'));
    C1 = C1(:,1);
    
    C2 =  textread(strcat('cb_',num2str(2^b_2), '.txt'));
    C2 = C2(:,1);
    
    %% Generating all combinations for the joint minimization
%     all_combs1 = combn(1:length(C1),b_ex);
%     all_combs2 = combn(1:length(C2),N - b_ex);

    % Sensing matrix 
    A_unnormalized = sqrt(1/N)*randn(N,M);
    s = sqrt(sum(A_unnormalized.^2));
    S = diag(1./s);
    A = A_unnormalized*S; % normalized sensing matrix A whose columns are normalized to unit-norm
        
    
    y_near = zeros(N,1);
        
    %% Start Monte-Carlo loop
    for mc=1:MC
            ind = randperm(M);
            x = zeros(M,1);
            x(ind(1:K)) = randn(K,1); % create a Gaussian K-sparse source. As an alternative, change it to "x(sel(1:K)) = 1" for zero-one K-sparse source
            y = A*x; % measurements
        
            % Locally reconstructed source using OMP algorithm (as an approximation to the MMSE estimator)
            [x_tilde I_tilde y_r_tilde] = OMP_func_CompEff(y,A,M,K);
            
            %% Nearesr neighbor coding (NNC)
            Y1 = y(1:b_ex)*ones(1,length(C1));
            Y2 = y(b_ex + 1:N)*ones(1,length(C2));
            % Note: the coefficeint 'sqrt(K/N)' below is to normalize 
            % a codepont (designed for a standard Gaussian N(0,1)) to each measurement entry (i.e., N(0,K/N)).
            [min_near1 ind_min_near1] = min(abs(Y1 - (sqrt(K/N)*ones(b_ex,1)*C1'))'); 
            [min_near2 ind_min_near2] = min(abs(Y2 - (sqrt(K/N)*ones(N - b_ex,1)*C2'))');
            y_near1 = sqrt(K/N).*C1(ind_min_near1);
            y_near2 = sqrt(K/N).*C2(ind_min_near2);
            y_near = [y_near1;y_near2]; % The quantized measurement vector based on NNC            
            %% Analysis by synthesis (AbS) algorithms. The AbS initializes with the NNC quantized vector. 
            
            % Sequential AbS algorithm  (Complexity: O(N*2^(Mr_x/N))))
            [y_hat_seq it_convergence_seq] = AbS_seq(K,A,b_ex,C1,C2,x_tilde,y_near,itr_max);
            
            % Non-sequential AbS algorithm (Complexity: O(N^2*2^(Mr_x/N)) )
%             [y_hat_nonseq it_convergence_nonseq] = AbS_nonseq(K,A,b_ex,C1,C2,x_tilde,y_near,itr_max);
            
            % Joint (optimal) minimization (Complexity: O(2^(Mr_x))))
%             [y_hat_joint] = AbS_joint(K,A,b_ex,C1,C2,all_combs1,all_combs2,x_tilde);
 
            %% Performance evaluation
            [x_hat_near I_hat_near y_r_hat_near] = OMP_func_CompEff(y_near,A,M,K); 
            
            [x_hat_abs_seq I_hat_abs_seq y_r_hat_abs_seq] = OMP_func_CompEff(y_hat_seq,A,M,K);
            
            
%             [x_hat_abs_nonseq I_hat_abs_nonseq y_r_hat_abs_nonseq] = OMP_func_CompEff(y_hat_nonseq,A,M,K)     
%             [x_hat_abs_joint I_hat_abs_joint y_r_hat_abs_joint] = OMP_func_CompEff(y_hat_joint,A,M,K);
            
             
             MSE_near(mc,n) = norm(x - x_hat_near,2).^2;
             MSE_abs_seq(mc,n) = norm(x - x_hat_abs_seq,2).^2;
%              MSE_abs_nonseq(mc,n) = norm(x - x_hat_abs_nonseq,2).^2;
%              MSE_abs_joint(mc,n) = norm(x - x_hat_abs_joint,2).^2;

             

            
    end % end-for Monte-Carlo loop
    
end % end-for N loop

% Normalized MSE (in dB)
NMSE_mean_near = 10*log10(mean(MSE_near)/K)
NMSE_mean_abs_seq = 10*log10(mean(MSE_abs_seq)/K)

% NMSE_mean_abs_nonseq = 10*log10(mean(MSE_abs_nonseq)/K)
% NMSE_mean_abs_joint = 10*log10(mean(MSE_abs_joint)/K)