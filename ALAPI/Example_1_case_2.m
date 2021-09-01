clearvars; close all; clc

%% Define the  parameterized imprecise probability models
Xinput.Dim = 2;                     % Dimension of random input X = [X1,X2]
Xinput.Dist = {'Normal','Normal'};  % Distribution type of X
Pinput.Dim = 4;                     % Dimension of distributional parameter P = [U1,S1,U2,S2]
Pinput.Dist = {'Uniform','Uniform','Uniform','Uniform'};  % Distribution type of P


Pinput.Bound = [-0.8  0.5  -0.8  0.5;
                 0.8   1.5   0.8 1.5];                   % Bound = [Lower;Upper]
            
%% Define the response function
% specity the paremeter
k = 4;
% specify the performance function
Per_Fun = @(x) min([3 + 0.1.*(x(:,1)-x(:,2)).^2 - (x(:,1)+x(:,2))./sqrt(2), 3 + 0.1*(x(:,1)-x(:,2)).^2 + (x(:,1)+x(:,2))./sqrt(2), (x(:,1) - x(:,2)) + k./sqrt(2), (x(:,2) - x(:,1)) + k./sqrt(2)],[],2);
            
            
%% Proposed ALAPI
LF = 'UPVC';
[AK_MCS_Result] = Active_learning_argument_probabilitic_integration(Xinput,Pinput,Per_Fun, LF);

%% NISS
[NISS_Result] = Non_intrusive_imprecise_stochastic_simulation(Xinput,Pinput,Per_Fun);

%% IASS
num_mcs = 1e6; 
[IASS_Result] = Imprecise_augmented_stochastic_simulation(Xinput,Pinput,Per_Fun,num_mcs);


% results

figure(1)
plot(0:length(AK_MCS_Result.UCOV)-1,AK_MCS_Result.UCOV,'b-','LineWidth',1.0)
xlabel('$\rm Number~of~adaptively~added~samples$','interpreter','latex','FontSize',12)
ylabel('$\kappa$','interpreter','latex','FontSize',12)
set(gca,'FontSize',12);
set(gca,'FontName','Timesnewroman');

%%
figure(2)
subplot(2,4,1)
plot(AK_MCS_Result.u1,AK_MCS_Result.Pf1,'b--','LineWidth',1.5)
hold on
plot(NISS_Result.u1,NISS_Result.Pf1,'g-.','LineWidth',1.5)
plot(AK_MCS_Result.u1,IASS_Result.Pf1,'r-','LineWidth',1.5)
xlabel('$\mu_1$','interpreter','latex','FontSize',10)
ylabel('$\hat{\mathcal{P}}_{f,{\rm RS},j}(\theta_j)$','interpreter','latex','FontSize',10)
grid on
set(gca,'FontSize',10);
set(gca,'FontName','Timesnewroman');

subplot(2,4,2)
plot(AK_MCS_Result.s1,AK_MCS_Result.Pf2,'b--','LineWidth',1.5)
hold on
plot(NISS_Result.s1,NISS_Result.Pf2,'g-.','LineWidth',1.5)
plot(AK_MCS_Result.s1,IASS_Result.Pf2,'r-','LineWidth',1.5)
xlabel('$\sigma_1$','interpreter','latex','FontSize',10)
grid on
set(gca,'FontSize',10);
set(gca,'FontName','Timesnewroman');

subplot(2,4,3)
plot(AK_MCS_Result.u2,AK_MCS_Result.Pf3,'b--','LineWidth',1.5)
hold on
plot(NISS_Result.u2,NISS_Result.Pf3,'g-.','LineWidth',1.5)
plot(AK_MCS_Result.u2,IASS_Result.Pf3,'r-','LineWidth',1.5)
xlabel('$\mu_2$','interpreter','latex','FontSize',10)
grid on
set(gca,'FontSize',10);
set(gca,'FontName','Timesnewroman');

subplot(2,4,4)
plot(AK_MCS_Result.s2,AK_MCS_Result.Pf4,'b--','LineWidth',1.5)
hold on
plot(NISS_Result.s2,NISS_Result.Pf4,'g-.','LineWidth',1.5)
plot(AK_MCS_Result.s2,IASS_Result.Pf4,'r-','LineWidth',1.5)
xlabel('$\sigma_2$','interpreter','latex','FontSize',10)
grid on
set(gca,'FontSize',10);
set(gca,'FontName','Timesnewroman');

subplot(2,4,5)
plot(AK_MCS_Result.u1,AK_MCS_Result.Pf1_UCOV,'b--','LineWidth',1.5)
hold on
plot(NISS_Result.u1,NISS_Result.Pf1_COV,'g-.','LineWidth',1.5)
plot(AK_MCS_Result.u1,IASS_Result.Pf1_COV,'r-','LineWidth',1.5)
xlabel('$\mu_1$','interpreter','latex','FontSize',10)
ylabel('$\overline{{\rm COV}}_{\mathcal{T}}[\hat{\mathcal{P}}_{f,{\rm RS},j}(\theta_j)]~{\rm or}~{\rm COV}[\hat{\mathcal{P}}_{f,{\rm RS},j}(\theta_j)]$','interpreter','latex','FontSize',10)
grid on
set(gca,'FontSize',10);
set(gca,'FontName','Timesnewroman');

subplot(2,4,6)
plot(AK_MCS_Result.s1,AK_MCS_Result.Pf2_UCOV,'b--','LineWidth',1.5)
hold on
plot(NISS_Result.s1,NISS_Result.Pf2_COV,'g-.','LineWidth',1.5)
plot(AK_MCS_Result.s1,IASS_Result.Pf2_COV,'r-','LineWidth',1.5)
xlabel('$\sigma_1$','interpreter','latex','FontSize',10)
grid on
set(gca,'FontSize',10);
set(gca,'FontName','Timesnewroman');

subplot(2,4,7)
plot(AK_MCS_Result.u2,AK_MCS_Result.Pf3_UCOV,'b--','LineWidth',1.5)
hold on
plot(NISS_Result.u2,NISS_Result.Pf3_COV,'g-.','LineWidth',1.5)
plot(AK_MCS_Result.u2,IASS_Result.Pf3_COV,'r-','LineWidth',1.5)
xlabel('$\mu_2$','interpreter','latex','FontSize',10)
grid on
set(gca,'FontSize',10);
set(gca,'FontName','Timesnewroman');

subplot(2,4,8)
plot(AK_MCS_Result.s2,AK_MCS_Result.Pf4_UCOV,'b--','LineWidth',1.5)
hold on
plot(NISS_Result.s2,NISS_Result.Pf4_COV,'g-.','LineWidth',1.5)
plot(AK_MCS_Result.s2,IASS_Result.Pf4_COV,'r-','LineWidth',1.5)
xlabel('$\sigma_2$','interpreter','latex','FontSize',10)
grid on

h = legend('ALAPI','NISS','IASS');
set(h,'Interpreter','latex','FontSize',10,'box','on','NumColumns',3,'Location','north')
set(gca,'FontSize',10);
set(gca,'FontName','Timesnewroman');


%% 
u1 = AK_MCS_Result.u1;
u2 = AK_MCS_Result.u2;
[T1,T2]=meshgrid(u1,u2);
figure(3)
subplot(1,2,1)
mesh(T1,T2,AK_MCS_Result.Pf13,'FaceColor','b','FaceAlpha',0.5);
hold on
mesh(T1,T2,IASS_Result.Pf13,'FaceColor','r','FaceAlpha',0.5);
h = legend('ALAPI','IASS');
set(h,'Interpreter','latex','FontSize',10,'box','on','NumColumns',2,'Location','north');
grid on
xl = xlabel('$\mu_1$','interpreter','latex','FontSize',10);
yl = ylabel('$\mu_2$','interpreter','latex','FontSize',10);
set(get(gca,'ylabel'),'rotation',-30,'VerticalAlignment','middle');
set(get(gca,'xlabel'),'rotation',30,'VerticalAlignment','middle');
zlabel('$\hat{\mathcal{P}}_{f,{\rm RS},13}(\mu_1,\mu_2)$','interpreter','latex','FontSize',10)
set(gca,'FontSize',10);
set(gca,'FontName','Timesnewroman');

subplot(1,2,2)
mesh(T1,T2,AK_MCS_Result.Pf13_UCOV,'FaceColor','b','FaceAlpha',0.5);
hold on
mesh(T1,T2,IASS_Result.Pf13_COV,'FaceColor','r','FaceAlpha',0.5);
grid on
xl = xlabel('$\mu_1$','interpreter','latex','FontSize',10);
yl = ylabel('$\mu_2$','interpreter','latex','FontSize',10);
set(get(gca,'ylabel'),'rotation',-30,'VerticalAlignment','middle');
set(get(gca,'xlabel'),'rotation',30,'VerticalAlignment','middle');
zlabel('$\overline{{\rm COV}}_{\mathcal{T}}[\hat{\mathcal{P}}_{f,{\rm RS},13}(\mu_1,\mu_2)]~{\rm or}~{\rm COV}[\hat{\mathcal{P}}_{f,{\rm RS},13}(\mu_1,\mu_2)]$','interpreter','latex','FontSize',10)
set(gca,'FontSize',10);
set(gca,'FontName','Timesnewroman');