clear all
close all

%% Import data set (predict und target)
result_layer_3 = readtable('version_6P1/regressor_layer_3.csv');
target =  readtable('version_6P1/regressor_target.csv');

result_layer_3 = table2array(result_layer_3);
result_layer_3(1,:) = [];

result_layer(3,:,:) = result_layer_3;

target = table2array(target);
target(1,:) = [];

%% plot
label = ["m2","m3","m4","k","alpha","beta"];

for i = 3
    figure(i)
    for j=1:6
        subplot(3,2,j)
%         result_layer(i, result_layer(i,:,j+1)<= 1.03 & result_layer(i,:,j+1) >= 0.97, j+1) = 1;
        scatter(target(:,j+1), result_layer(i,:,j+1))
        grid on
        hold on
        reference_curve = linspace(min(result_layer(i,:,j+1)),max(result_layer(i,:,j+1)));
        plot(reference_curve, reference_curve, 'LineWidth', 2)
        xlim([min(result_layer(i,:,j+1)), max(result_layer(i,:,j+1))])
        ylim([min(target(:,j+1)), max(target(:,j+1))])
        xlabel('target')
        ylabel('prediction')
        title(label(1,j))
    end
end
% sgtitle('version_6P1 1P')
sgtitle('gerundete 1P  Daten in 7P Modell')

for i = 3
    figure(3+i)
    for j=1:6
        subplot(3,2,j)
        scatter(target(1:200,1),target(1:200,j+1),'*')
        grid on
        hold on
        scatter(result_layer(i,1:200,1),result_layer(i,1:200,j+1))
        xlabel('examples')
        ylabel(label(1,j))
        legend('true', 'predict')
    end
end
sgtitle('gerundete 1P Daten in 7P Modell')


