C_list = [-0.5];
features_list = [20];
range = 0.5; overlap = 0.0; datasize = 750; train_ratio = 2/3;
trials = 1;
train_size = datasize*2*train_ratio;

%% generate the data.
dataset = GenerateData(datasize,range,overlap);
save('dataset.mat','dataset');

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

%% estimate the sigma parameter
sigma = sqrt(sum(sum(K)))/size(K,1);
K = exp(-K/2/sigma^2);

%% sparse random features method.
spa_results = [];
%% outer loop sweeps different number of features
for NoFeatures = features_list
    %% middle loop runs repeated trials, each trial use the same group of random features
    for kdx = 1:trials
        %% generate approximate data.
        Xtil = RFF(dataset(:,1:2),NoFeatures,sigma);
        XtilTrain = Xtil(1:2*datasize*train_ratio,:);
        XtilTest = Xtil(2*datasize*train_ratio+1:2*datasize,:);
        %% inner loops sweeps different C's
        for C_power = C_list
            Ctil = 10^C_power;
            %% hinge loss using L1 regularization
            model = fitclinear(XtilTrain,y_train,'Lambda',1/Ctil/train_size,'Regularization','lasso');
            [~,Scores] = predict(model,XtilTest);
            accuracytil1 = mean(Scores(:,2).*y_test>0);
            sparsity = sum([model.Beta~=0]);
            Output = [accuracytil1,sparsity,C_power,NoFeatures,kdx];
            spa_results = [spa_results;Output];
        end;
    end;
    disp(NoFeatures)
end;

%% compute the Xtil.*y and columns extracted by the model
Xy = XtilTrain.*(y_train*ones(1,size(XtilTrain,2)));
nonzerocols = model.Beta~=0;
Xy_ext = Xy.*(ones(size(XtilTrain,1),1)*nonzerocols');

%% plot
fig = figure(2);
set(fig,'PaperPositionMode','auto','Units','inches','Position',[0 0 6 4]);
imagesc(Xy);
print('colormatrix','-depsc');
fig = figure(3);
set(fig,'PaperPositionMode','auto','Units','inches','Position',[0 0 6 4]);
imagesc(Xy_ext);
print('colormatrix_ext','-depsc');
