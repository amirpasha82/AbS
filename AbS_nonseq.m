% Implementing non-sequential AbS quantization as reported in the paper 'Analysis-by-synthesis Quantization 
% for Compressed Sensing Measurements' by Amirpasha Shirazinia, Saikat
% Chatterjee and Mikael Skoglund submitted to IEEE Trans. SP, 2013.

% Written by: Amirpasha Shirazinia, Communication Theory Lab, KTH
% Email: amishi@kth.se
% Created: May 18, 2013

function[y_hat it_converge idxofmin_2] = AbS_nonseq(K,A,b_ex,C1,C2,x_tilde,y_near,itr_max)
        
[N M] = size(A);
Delta_MSE = 1;
it = 1;
temp_it = zeros(1,itr_max);
y_hat = y_near;        

while Delta_MSE > 1e-12
         set1 = 1:b_ex;
         while ~isempty(set1)
            MSE_abs_alter_1 = zeros(length(set1),length(C1));
            for n1=1:length(set1)
                temp_1 = y_hat;
                for l1=1:length(C1)
                    temp_1(n1) = C1(l1)*sqrt(K/N);
                    [x_hat1 , ~, ~] = OMP_func_CompEff(temp_1,A,M,K);
                    MSE_abs_alter_1(n1,l1) = norm(x_tilde - x_hat1,2);
                end
            end
            [~,idxofmin_1] = min(MSE_abs_alter_1(:)); % find the row and column of the minimum element of a matrix
            [n_star_1,i_star_1] = ind2sub(size(MSE_abs_alter_1),idxofmin_1);
            y_hat(n_star_1) = C1(i_star_1)*sqrt(K/N);
            set1 = setdiff(1:length(set1),n_star_1);
         end 
         
         set2 = 1:N-b_ex;
         while ~isempty(set2)
            MSE_abs_alter_2 = zeros(length(set2),length(C2));
            for n2=1:length(set2)
                temp_2 = y_hat;
                for l2=1:length(C2)
                    temp_2(b_ex + n2) = C2(l2)*sqrt(K/N);
                    [x_hat2, ~,  ] = OMP_func_CompEff(temp_2,A,M,K);
                    MSE_abs_alter_2(n2,l2) = norm(x_tilde - x_hat2,2);
                end
            end
            [~,idxofmin_2] = min(MSE_abs_alter_2(:)); % find the row and column of the minimum element of a matrix
            [n_star_2,i_star_2] = ind2sub(size(MSE_abs_alter_2),idxofmin_2);
            y_hat(b_ex + n_star_2) = C2(i_star_2)*sqrt(K/N);
            set2 = setdiff(1:length(set2),n_star_2);
             
         end 
            
     [x_hat , ~, ~] = OMP_func_CompEff(y_hat,A,M,K);
     temp_it(it+1) = norm(x_hat - x_tilde,2);
     Delta_MSE = abs(temp_it(it+1) - temp_it(it));
     it = it + 1;
end
it_converge = it - 1;
end