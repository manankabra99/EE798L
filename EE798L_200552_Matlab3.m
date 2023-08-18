clc; clear all; close all;



% PART A --- Generating the data
% Initialising the given gmm parameters.
n = 500;
mu_gen1 = [3, 3];
mu_gen2 = [1, -3];
mu_gen = [mu_gen1; mu_gen2];
sig_gen1 = [1, 0; 0, 2];
sig_gen2 = [2, 0; 0, 1];
sig_gen = cat(3, sig_gen1, sig_gen2);
pi_gen = [0.8, 0.2];


% Generating the data
gmm = gmdistribution(mu_gen, sig_gen, pi_gen);
X = random(gmm, n);

figure(1);
gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(gmm,[x0 y0]),x,y);
fcontour(gmPDF,[-3 7 -5 7]);
hold on;
scatter(X(:,1),X(:,2),7,'.');
title('1. Plot of Generated data with given Gaussians');
hold off;


% PART B --- Writing the EM algorithm
k = 2;
mu_k = zeros(k, 2);
sig_k = zeros(k, 2, 2);
pi_k = zeros(k, 1);
q = ones(n, k)/2;

sig1 = zeros(2, 2);
sig2 = zeros(2, 2);
mu1 = zeros(1, 2);
mu2 = zeros(1, 2);
pi1 = 0;
pi2 = 0;
for i = 1:n
    q(i, 1) = 0.45;
    q(i, 2) = 0.55;
end
while true
    % STEP 1
    S = sum(q, 1);
    pi_k = S/n;
    
    temp1 = zeros(1, 2);
    temp2 = zeros(1, 2);
    for i = 1:n
        temp1 = temp1 + X(i, :)*q(i, 1);
        temp2 = temp2 + X(i, :)*q(i, 2);
    end
    mu_k(1, :) = (temp1')/S(1);
    mu_k(2, :) = (temp2')/S(2);
    %disp(mu_k);

    temp3 = zeros(2, 2);
    temp4 = zeros(2, 2);
    for i = 1:n
        temp3 = temp3 + q(i, 1)*(((X(i, :) - mu_k(1, :))')*(X(i, :) - mu_k(1, :)));
        temp4 = temp4 + q(i, 2)*(((X(i, :) - mu_k(2, :))')*(X(i, :) - mu_k(2, :)));
    end
    sig_k(1, :, :) = temp3/S(1);
    sig_k(2, :, :) = temp4/S(2);
    %disp(sig_k);


    sig1 = reshape(sig_k(1, :, :), 2, 2);
    sig2 = reshape(sig_k(2, :, :), 2, 2);
    mu1 = reshape(mu_k(1, :), 1, 2);
    mu2 = reshape(mu_k(2, :), 1, 2);
    pi1 = pi_k(1);
    pi2 = pi_k(2);
    % STEP 2
    q_new = zeros(n, k);
    temp = zeros(500);
    for i = 1:500
        temp(i) = temp(i) + pi_k(1)*(gauss(X(i, :), mu_gen1, sig_gen1)) + pi_k(2)*(gauss(X(i, :), mu_gen2, sig_gen2));
    end
    for i = 1:500
        q_new(i, 1) = pi_k(1)*(gauss(X(i, :), mu_gen1, sig_gen1))/(temp(i));
        q_new(i, 2) = pi_k(2)*(gauss(X(i, :), mu_gen2, sig_gen2))/(temp(i));
    end
    
    diff = 0;
    for i = 1:500
        diff = diff + (q(i, 1) - q_new(i, 1))*(q(i, 1) - q_new(i, 1));
    end

    if diff < 0.001
        break;
    end
    q = q_new;
    %break;
end

disp("Parameters of the First Gaussian are - ");
disp("Pi1 is - ");
disp(pi1);
disp("Mu1 is - ");
disp(mu1);
disp("Sig1 is - ");
disp(sig1);

disp("Parameters of the Second Gaussian are - ");
disp("Pi2 is - ");
disp(pi2);
disp("Mu2 is - ");
disp(mu2);
disp("Sig2 is - ");
disp(sig2);


%PART C -- Reassigning points


mu = [mu1; mu2];
sig = cat(3, sig1, sig2);
pi = [pi1, pi2];
%gmm = gmdistribution(mu, sig, pi);

c1 = zeros(500, 2);
c2 = zeros(500, 2);
s1 = 1;
s2 = 1;

for i = 1:500
    val1 = gauss(X(i, :), mu_gen1, sig_gen1);
    val2 = gauss(X(i, :), mu_gen2, sig_gen2);
    if val1 >= val2
        c1(s1, :) = X(i, :);
        s1 = s1 + 1;
    end
    if val1 < val2
        c2(s2, :) = X(i, :);
        s2 = s2 + 1;
    end
end

s1 = s1 - 1;
s2 = s2 - 1;

X1 = zeros(s1, 2);
for i = 1:s1
    X1(i, :) = c1(i, :);
end
X2 = zeros(s2, 2);
for i = 1:s2
    X2(i, :) = c2(i, :);
end


figure(2);
gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(gmm,[x0 y0]),x,y);
fcontour(gmPDF,[-3 7 -5 7]);
hold on;
scatter(X1(:,1),X1(:,2),7,'x');
hold on;
scatter(X2(:,1),X2(:,2),7,'+')
title('3. Plot of data after Assigning to clusters');

% Define Gaussian function
function x = gauss(xn ,mu, sigma)
    temp = (xn - mu)*(inv(sigma))*(xn - mu)';
    x = exp(-temp);
    x = x / (2*pi);
    d = det(sigma);
    x = x / sqrt(d);
end