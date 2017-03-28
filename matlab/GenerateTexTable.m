load('acc_results.mat');
load('mean_results.mat');
load('err_results.mat');

all_results(:,1) = acc_results(:,1);
for idx = [2:1:7]
    all_results(:,idx) = mean_results(idx-1,:);
    err(:,idx-1) = err_results(idx-1,:);
end;
dlmwrite('mean.tex',all_results,'delimiter','&','precision','%.2f');
dlmwrite('err.tex',err,'delimiter','&','precision','%.2e');
