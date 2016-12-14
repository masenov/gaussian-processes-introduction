interval_size = 10;
x = [0:0.01:interval_size];
x = [-interval_size:0.01:interval_size];
plot(x,gaussian(x,0,5))
hold on;
plot(x,gaussian(x,0,3))
hold on;
plot(x,gaussian(x,0,1))
hold on;
plot(x,gaussian(x,2,1));
xlabel('x'); ylabel('Probability density');
title('Gaussian distribution')
l= legend(['$\mu=0,\sigma^2=5$';'$\mu=0,\sigma^2=1$';'$\mu=0,\sigma^2=3$';'$\mu=2,\sigma^2=1$']);

set(l,'Interpreter','Latex');
saveas(gcf,'../pictures/gaussian.png');
%%
close all;
mu = [0 0];
Sigma = [.25 .3; .3 1];
x1 = -3:.2:3; x2 = -3:.2:3;
[X1,X2] = meshgrid(x1,x2);
F = gaussian_multivariate([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));
surf(x1,x2,F);
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
axis([-3 3 -3 3 0 .4])
xlabel('x1'); ylabel('x2'); zlabel('Probability Density');
%%
mu = [0 0];
Sigma = [.25 .3; .3 1];
x1 = -3:.2:3; x2 = -3:.2:3;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));

mvncdf([0 0],[1 1],mu,Sigma);
contour(x1,x2,F,[.0001 .001 .01 .05:.1:.95 .99 .999 .9999]);
xlabel('x'); ylabel('y');
