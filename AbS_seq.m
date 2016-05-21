% Implementing sequential AbS quantization as reported in the paper 'Analysis-by-synthesis Quantization 
% for Compressed Sensing Measurements' by Amirpasha Shirazinia, Saikat
% Chatterjee and Mikael Skoglund submitted to IEEE Trans. SP, 2013.

% Written by: Amirpasha Shirazinia, Communication Theory Lab, KTH
% Email: amishi@kth.se
% Created: May 18, 2013

function[y_hat it_converge indi] = AbS_seq(K,A,b_ex,C1,C2,x_tilde,y_near,itr_max)        
        [N M] = size(A);
        Delta_MSE = 1;
        it = 1;
        temp = zeros(1,itr_max);
        y_hat = y_near;
        while Delta_MSE > 1e-12
            indi = zeros(1,N);
            for k=1:N
                MSE_abs1 = zeros(1,length(C1));
                MSE_abs2 = zeros(1,length(C2));
                if k <= b_ex
                    for l1=1:length(C1)
                        y_hat(k) = C1(l1)*sqrt(K/N);
                        [x_hat1 , ~, ~] = OMP_func_CompEff(y_hat,A,M,K);
                        MSE_abs1(l1) = norm(x_tilde - x_hat1,2);
                    end
                    [~, ind_abs_min1] = min(MSE_abs1);
                    y_hat(k) = C1(ind_abs_min1)*sqrt(K/N);
                    
                else
                    for l2=1:length(C2)
                        y_hat(k) = C2(l2)*sqrt(K/N);
                        [x_hat2 , ~, ~] = OMP_func_CompEff(y_hat,A,M,K);
                        MSE_abs2(l2) = norm(x_tilde - x_hat2,2);
                    end
                    [~, ind_abs_min2] = min(MSE_abs2);            
                    y_hat(k) = C2(ind_abs_min2)*sqrt(K/N);
                    indi = ind_abs_min2;
                end
                
            end
            temp(it+1) = norm(x_tilde - x_hat2,2);
            Delta_MSE = abs(temp(it+1) - temp(it));
            it = it + 1;
        end 
        it_converge = it - 1;
end
