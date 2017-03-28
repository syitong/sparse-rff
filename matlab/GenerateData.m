function Data = GenerateData(SampleSize,Range,Overlap)

Rho1 = -rand(SampleSize,1)*Range+1+Range*Overlap;
Theta1 = rand(SampleSize,1)*2*pi;
X1 = [Rho1.*cos(Theta1),Rho1.*sin(Theta1)];
Y1 = -ones(SampleSize,1);
Rho2 = rand(SampleSize,1)*Range+1-Range*Overlap;
Theta2 = rand(SampleSize,1)*2*pi;
X2 = [Rho2.*cos(Theta2),Rho2.*sin(Theta2)];
Y2 = ones(SampleSize,1);
Data = [X1,Y1;X2,Y2];
Data = Data(randperm(2*SampleSize),:);