 clc; clear; close all;
%% ===============================
% PART 1: Stock Price Prediction (Linear Regression)
% ===============================

data = readtable('stock_data.csv');

% Convert Date column (if it's in string format)
data.Date = datetime(data.Date, 'InputFormat','yyyy-MM-dd');

% Sort by date
data = sortrows(data, 'Date');

disp('Column names:');
disp(data.Properties.VariableNames);

% Define Target
closingPrices = data.Close;
dates = data.Date;
days = (1:length(closingPrices))';

% Train/Test Split
splitRatio = 0.8;
splitIndex = floor(splitRatio * length(closingPrices));

X_train = days(1:splitIndex);
y_train = closingPrices(1:splitIndex);

X_test = days(splitIndex+1:end);
y_test = closingPrices(splitIndex+1:end);

% Train Linear Regression Model
mdl = fitlm(X_train, y_train);

% Predict
y_pred = predict(mdl, X_test);

% Plot Results
figure;
plot(dates, closingPrices, 'b', 'LineWidth', 1.5); hold on;
plot(dates(splitIndex+1:end), y_pred, 'r--', 'LineWidth', 2);
legend('Actual Prices', 'Predicted Prices');
xlabel('Date'); ylabel('Closing Price');
title('Stock Market Prediction using Linear Regression');
grid on;

% Evaluate Model
mse = mean((y_test - y_pred).^2);
fprintf('\n--- Linear Regression Results ---\n');
fprintf('Mean Squared Error: %.2f\n', mse);

%% ===============================
% PART 2: Monte Carlo Simulation
% ===============================

% Parameters (use last actual price as starting point)

S0 = closingPrices(end); 
mu = 0.10;   % Expected return (10%)
sigma = 0.2; % Volatility (20%)
T = 1;       % Time horizon (1 year)
N = 252;     % Steps (daily ~ 252 trading days)
M = 1000;    % Number of simulations

dt = T/N; 
t = linspace(0, T, N);

% Simulations
S = zeros(N, M); 
S(1,:) = S0;

for j = 1:M
    for i = 2:N
        Z = randn;
        S(i,j) = S(i-1,j) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z);
    end
end
% Plot Sample Paths
figure;
plot(t, S(:,1:20));
xlabel('Time (Years)'); ylabel('Stock Price');
title('Monte Carlo Simulation of Stock Prices');

% Distribution at Final Time
final_prices = S(end,:);
figure;
histogram(final_prices, 30);
xlabel('Final Stock Price'); ylabel('Frequency');
title('Distribution of Final Stock Prices');

% Probability of exceeding target
target_price = S0 * 1.2; % 20% growth target
prob = mean(final_prices > target_price);

fprintf('\n--- Monte Carlo Simulation Results ---\n');
fprintf('Initial Stock Price: %.2f\n', S0);
fprintf('Expected Stock Price after %.1f year: %.2f\n', T, mean(final_prices));
fprintf('Probability Stock > %.2f: %.2f%%\n', target_price, prob*100); 

