% Computationally efficient OMP algorithm that computes Pseduo-inverse in a
% recursive manner

% Implementing the Orthogonal Matching Pursuit Algorithm as reported in the paper
% of 'Gradient Pursuits' by Thomas Blumensath and Mike E. Davies
% Written following the OMP Algorithm Section II of the authors' paper 
% in IEEE Trans SP
% Paper: Gradient pursuits; IEEE Trans SP, June 2008

% Signal model: y=AX+W. In the authors' paper, main consideration is 'W' is
% a zero vector (i.e. no measurement noise)

% Written by: Saikat Chatterjee of KTH
% Email: saikatchatt@gmail.com
% Created: 15'th September 2010


function [xp I y_r]=OMP_func_CompEff(y,A,N,K_Max)

% Input
% N <- Dimension of the input data 'X' that is K-sparse (either strictly or weakly)
% y <- 'M' dimensional measurement data (an instance of Y where Y=AX+W)
% A <- Measument matrix of dimension M*N

% Output
% xp -> estimate of 'X' 
% I -> Support set of xp_SP
% y_r -> residue of the mismatch (please see the paper to understand)


% Initialization
x0 = A'*y;   % Initial guess (min energy)
temp1=abs(x0);
[temp2 I0]=max(temp1);
% [I0 I0_compliment]=support_set_eval_max(x0);   % Support set evaluation for maximum amplitude

P_0= (A(:,I0))'*(A(:,I0));
P_0_inv=1/P_0;  % Inverse calculation (Note its a scalar)
g_0=(A(:,I0))'*y;

x_temp=P_0_inv*g_0;

y_residue0= y - A(:,I0)*x_temp;

% y_residue0=residue_for_OMP(y,A,I0,N);

y_residue_old=y_residue0;
I_Old=I0;
A_Old=A(:,I_Old);
P_Old_inv=P_0_inv;
g_Old=g_0;
x_Old=x_temp;



for k=2:K_Max
     
    % Finding largest amplitude coeff through residue correlation matching
    % Mathced filtering
    temp1=abs(A'*y_residue_old);
    [temp2 I_k]=max(temp1);
        
    % Forming the support set
    I_New=[I_Old I_k];
    A_New=A(:,I_New);
    
    % Calculating the inverse of P_New in recursive way
    gamma=(A(:,I_k))'*A(:,I_k);
    
    q_k = A_Old'*A(:,I_k);
    dummy_k= P_Old_inv * q_k;
    beta_k= gamma - (q_k'*dummy_k);
    beta_k_inv=1/beta_k;
    
    A_block = P_Old_inv + (beta_k_inv * (dummy_k * dummy_k'));
    B_column= - (beta_k_inv * dummy_k);
    C_row=B_column';
    D_scalar=beta_k_inv;
    
    P_New_inv=[A_block B_column; C_row D_scalar];
    
    % Calculating g_New in recursive way
    g_New=[g_Old; A(:,I_k)'*y];
    
    % Calculating the Pseduo-inverse for the set of indices I_New
    x_New=P_New_inv*g_New;
    
    % Calculating the orthogonal projection residual 
    y_residue_New=y - A_New*x_New;
    
    
    if ( (norm(y_residue_New) >= norm(y_residue_old)) )   % Stopping criterion
        I_New=I_Old;
        x_New=x_Old;
        y_residue_New=y_residue_old;
        break;
    else
        I_Old=I_New;
        x_Old=x_New;
        y_residue_old=y_residue_New;
        A_Old=A_New;
        P_Old_inv=P_New_inv;
        g_Old=g_New;
    end
    
    
    
end
    
xp_OMP=zeros(N,1);
xp_OMP(I_New)=x_New;
xp=xp_OMP;
y_r=y_residue_New;
I=I_New;
