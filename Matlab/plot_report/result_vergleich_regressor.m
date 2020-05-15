clear all
%% Import data set (predict und target)
result_layer_1 = readtable('PmitT/regressor_layer_1_pred.csv');
result_layer_2 = readtable('PmitT/regressor_layer_2_pred.csv');
result_layer_3 = readtable('PmitT/regressor_layer_3_pred.csv');
target =  readtable('PmitT/regressor_target.csv');

result_layer_1 = table2array(result_layer_1);
result_layer_1(1,:) = [];

result_layer_2 = table2array(result_layer_2);
result_layer_2(1,:) = [];

result_layer_3 = table2array(result_layer_3);
result_layer_3(1,:) = [];

result_layer(1,:,:) = result_layer_1;
result_layer(2,:,:) = result_layer_2;
result_layer(3,:,:) = result_layer_3;

target = table2array(target);
target(1,:) = [];

%% plot
label = ["m2","m3","m4","k","alpha","beta"];
for i = 3
    figure(i)
    for j=1:6
        subplot(3,2,j)
        scatter(result_layer(i,:,j+1),target(:,j+1))
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

for i = 3
    figure(3+i)
    for j=1:6
        subplot(3,2,j)
        scatter(target(:,1),target(:,j+1),'*')
        grid on
        hold on
        scatter(result_layer(i,:,1),result_layer(i,:,j+1))
        xlabel('examples')
        ylabel(label(1,j))
        legend('true', 'predict')
    end
end
