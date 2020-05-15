%% Plot for 4.3.3
x = linspace(-1, 1);
y_relu = ReLU(x);
x = linspace(-10, 10);
y_sigmoid = sigmoid(x);

figure(1)
subplot(1,2,1)
plot(x, y_relu, 'LineWidth', 2, 'Color', 'r')
grid on
hold on
title('ReLU function')
xlabel('Intermediate variables z')
ylabel('acitvation')
ylim([-0.1, 1])

subplot(1,2,2)
plot(x, y_sigmoid, 'LineWidth', 2)
grid on
hold on
title('Sigmoid function')
xlabel('Intermediate variables z')
ylabel('acitvation')
ylim([-0.1, 1])

%% Plot for 4.4.1
% Underfitting
x = linspace(-1, 2);
y = x.^2;
x_rand = rand(1,100);
y_rand = rand(1,100);

figure(2)
plot(x+x_rand, y+y_rand,'*')
hold on
x = linspace(-1, 3);
plot(x, 1.5*x, 'LineWidth', 2)
title('Underfitting')
legend('Target', 'Proception')

%% Plot for 4.4.2
% Overfitting
x = linspace(-1, 2);
y = x.^2;
x_rand = rand(1,100);
y_rand = rand(1,100);

p_fit = polyfit(x+x_rand, y+y_rand, 20);
y_fit = polyval(p_fit, x);
figure(3)
subplot(1,2,1)
plot(x+x_rand, y+y_rand,'*')
hold on
x = linspace(-1, 3);
plot(x, y_fit, 'LineWidth', 2)
title('Overfitting in Training')
legend('Target', 'Proception')
grid on
xlim([-1 3])
ylim([0 3])

subplot(1,2,2)
plot(x+4*x_rand-2*y_rand, 1.5*y+0.2*y_rand+2.1*x_rand,'*')
hold on
x = linspace(-1, 3);
plot(x, y_fit, 'LineWidth', 2)
title('Overfitting in Test')
legend('Target', 'Proception')
grid on
xlim([-1 3])
ylim([0 3])