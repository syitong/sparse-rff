%% initialization.
C_list = [-3:0.5:4];
range = 0.5; overlap = 0.0; datasize = 750; train_ratio = 2/3;
features_list = [2,5,10,20,40,100];
trials = 50;
train_size = datasize*2*train_ratio;
rng(0);
%% generate the data.
dataset = GenerateData(datasize,range,overlap);
%% load the existed data.
%load('Data1.mat');
%% plot samples
fig = figure(1);
set(fig,'PaperPositionMode','auto','Units','inches','Position',[0 0 4 4]);
index = randperm(datasize*2,50);
pos_index = index(dataset(index,3)>0);
neg_index = index(dataset(index,3)<0);
plot(dataset(pos_index,1),dataset(pos_index,2),'ro');hold on;
plot(dataset(neg_index,1),dataset(neg_index,2),'bx');hold on;
ezpolar('1');
xlim([-1.5 1.5]);ylim([-1.5 1.5]);
daspect([1 1 1]);
print('samples','-depsc');
%% split into training and testing data sets.
x_train = dataset(1:2*datasize*train_ratio,1:2);
y_train = dataset(1:2*datasize*train_ratio,3);
x_test = dataset(2*datasize*train_ratio+1:2*datasize,1:2);
y_test = dataset(2*datasize*train_ratio+1:2*datasize,3);
%% calculate the accurate kernel matrix.
for idx = 1:size(x_train,1)
    for jdx = 1:size(x_train,1)
        K(idx,jdx) = norm(x_train(idx,:)-x_train(jdx,:))^2;
    end;
end;
sigma = sqrt(sum(sum(K)))/size(K,1);
K = exp(-K/2/sigma^2);
%% reproduce the result, when load previous data.
%rng(0);
%% train the accurate SVM model.
acc_results = [];
for idx = [1:size(C_list,2)]
    C_power = C_list(idx);
    C = 10^C_power;
    %% accurate svm
    model = fitcsvm(x_train,y_train,'BoxConstraint',C,'KernelFunction','rbf','KernelScale',sqrt(2)*sigma);
    [~,Scores] = predict(model,x_test);
    accuracy = mean(Scores(:,2).*y_test>0);
    what = sqrt(model.Alpha'*((y_train(model.IsSupportVector>0)*y_train(model.IsSupportVector>0)').*K(model.IsSupportVector>0,model.IsSupportVector>0))*model.Alpha);
    acc_results(idx,:) = [accuracy,what,C_power];
    disp(C_power);
end;
%% random fourier features method.
app_results = [];
spa_results = [];
%% outer loop sweeps different number of features
for NoFeatures = features_list
    %% middle loop runs repeated trials, each trial use the same group of random features
    for kdx = 1:trials
        %% generate approximate data.
        Xtil = RFF(dataset(:,1:2),NoFeatures,sigma);
        XtilTrain = Xtil(1:2*datasize*train_ratio,:);
        XtilTest = Xtil(2*datasize*train_ratio+1:2*datasize,:);
        %% calculate approximate kernel matrix.
        Ktil = XtilTrain*XtilTrain';
        %% inner loops sweeps different C's
        for C_power = C_list
            Ctil = 10^C_power;
            model = fitclinear(XtilTrain,y_train,'Lambda',1/Ctil/train_size);
            [~,Scores] = predict(model,XtilTest);
            accuracytil = mean(Scores(:,2).*y_test>0);
            sparsity = sum([model.Beta!=0]);
            Output = [accuracytil,sparsity,C_power,NoFeatures,kdx];
            app_results = [app_results;Output];
        end;
        %% random features using L1 regularization
        for C_power = C_list
            Ctil = 10^C_power;
            model = fitclinear(XtilTrain,y_train,'Lambda',1/Ctil/train_size,'Regularization','lasso');
            [~,Scores] = predict(model,XtilTest);
            accuracytil1 = mean(Scores(:,2).*y_test>0);
            sparsity = sum([model.Beta!=0]);
            Output = [accuracytil1,sparsity,C_power,NoFeatures,kdx];
            spa_results = [spa_results;Output];
        end;
    end;
    disp(NoFeatures)
end;
%% generate the dataset for plot
for idx = [0:size(features_list,2)-1]
    for jdx = [1:size(C_list,2)]
        for kdx = [0:trials-1]
            x(kdx+1) = app_results(idx*150+kdx*15+jdx,1);
        end;
        mean_results(idx+1,jdx) = mean(x);
        err_results(idx+1,jdx) = std(x);
    end;
end;
%% plot
fig = figure(2);
set(fig,'PaperPositionMode','auto','Units','inches','Position',[0 0 6 4]);
plot(C_list,acc_results(:,1),'-ro','MarkerSize',8,'LineWidth',1);hold on;
errorbar(C_list,mean_results(3,:),err_results(3,:),'--bs','MarkerSize',8,'LineWidth',1);
errorbar(C_list,mean_results(6,:),err_results(6,:),'--gx','MarkerSize',8,'LineWidth',1);
xlabel('$\log(C)$','interpreter','latex','FontSize',10);ylabel('Classification Accuracy','FontSize',10);
legend({'accurate model','10 features','100 features'},'FontSize',10,'Location','southeast');
xlim([min(C_list) max(C_list)]);
print('results','-depsc');
save('dataset.mat','dataset');
save('acc_results.mat','acc_results');
save('app_results.mat','app_results');
save('mean_results.mat','mean_results');
save('err_results.mat','err_results');
