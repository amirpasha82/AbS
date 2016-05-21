% Implementing joint (optimal) quantization as reported in the paper 'Analysis-by-synthesis Quantization 
% for Compressed Sensing Measurements' by Amirpasha Shirazinia, Saikat
% Chatterjee and Mikael Skoglund submitted to IEEE Trans. SP, 2013.

% Written by: Amirpasha Shirazinia, Communication Theory Lab, KTH
% Email: amishi@kth.se
% Created: May 18, 2013


function[y_joint] = AbS_joint(K,A,b_ex,C1,C2,all_combs1,all_combs2,x_tilde)
    
    [N M] = size(A);
    len_all_combs = length(all_combs1) + length(all_combs2);
    MSE_abs_joint_local = zeros(1,len_all_combs);
    Y_joint_1 = zeros(b_ex,length(all_combs1));
    Y_joint_2 = zeros(N - b_ex,length(all_combs2));
    Y_joint = zeros(N,len_all_combs);
    
    a = 0;
    for a2 = 1:length(all_combs2)
        Y_joint_2(:,a2) = C2(all_combs2(a2,:))*sqrt(K/N);
        if b_ex == 0
            a = a2;
            Y_joint(:,a) = Y_joint_2(:,a2);
            [x_hat_joint , ~, ~] = OMP_func_CompEff(Y_joint(:,a),A,M,K);
            MSE_abs_joint_local(a) = norm(x_tilde - x_hat_joint,2);
        else
            for a1 = 1:length(all_combs1)
                Y_joint_1(:,a1) = C1(all_combs1(a1,:))*sqrt(K/N);
                a = a + 1;
                Y_joint(:,a) = [Y_joint_1(:,a1);Y_joint_2(:,a2)];
                [x_hat_joint , ~, ~] = OMP_func_CompEff(Y_joint(:,a),A,M,K);
                MSE_abs_joint_local(a) = norm(x_tilde - x_hat_joint,2);
            end
        end
    end
    [~, ind_min] = min(MSE_abs_joint_local);
    y_joint = Y_joint(:,ind_min);
end
