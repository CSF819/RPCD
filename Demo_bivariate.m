%% demo_bivariate.m
% Self-contained RCR demo for Pattern Recognition submission
% Outputs: Fig1a_scatter, Fig1b_condmean, Fig1c_rcr in PDF/PNG/SVG
% No grid lines | Times New Roman | Global std RCR | Correct direction

clc; clear; close all;
rng(2025); n = 1000;
X = rand(n,1); Y = X.^2 + 0.1*randn(n,1);

% === 运行 RCR ===
[ind,delta,VRCR_x,VRCR_y,RCR_x,RCR_y,E_X_given_Y,E_Y_given_X,grid_Y,grid_X,X_norm,Y_norm] = ...
    RCR_direction_continuous(X,Y);

% 平滑曲线准备
win = max(5, round(n/20));
[XS, idxX] = sort(X_norm); RCR_y_sorted = RCR_y(idxX);
[YS, idxY] = sort(Y_norm); RCR_x_sorted = RCR_x(idxY);

% === 全局字体设置 ===
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', 10);
set(0, 'DefaultTextFontSize', 10);

colors = struct('blue', [0, 0.4470, 0.7410], 'red', [0.8500, 0.3250, 0.0980], 'black', [0,0,0]);

% ==============================================================
% Figure (a): Scatter + True Curve
% ==============================================================
fig_a = figure('Color','w', 'Position',[100 100 420 340]);
scatter(X, Y, 36, colors.blue, 'filled', 'MarkerFaceAlpha', 0.6); hold on;
xx = linspace(0,1,200); plot(xx, xx.^2, 'Color', colors.red, 'LineWidth', 2.0);
xlabel('X'); ylabel('Y');
title('Synthetic data: Y = X^2 + \epsilon', 'FontWeight','bold', 'FontSize',12);
legend({'Samples', 'True Y = X^2'}, 'Location','northwest', 'Box','on', 'FontSize',10);
set(gca, 'FontName','Times New Roman', 'LineWidth',1.0, 'Box','on');
xlim([0 1]); ylim([0 1.2]); grid off;
tightfig(fig_a);
export_fig(fig_a, 'Fig1a_scatter');

% ==============================================================
% Figure (b): Conditional Means
% ==============================================================
fig_b = figure('Color','w', 'Position',[100 100 420 340]);
plot(grid_X, E_Y_given_X, 'Color', colors.blue, 'LineWidth', 2.5); hold on;
plot(grid_Y, E_X_given_Y, '--', 'Color', colors.black, 'LineWidth', 2.0);
xlabel('Normalized variable'); ylabel('Conditional mean');
title('Empirical E[Y|X] and E[X|Y] (20 bins)', 'FontWeight','bold', 'FontSize',12);
legend({'E[Y|X]', 'E[X|Y]'}, 'Location','southeast', 'Box','on', 'FontSize',10);
set(gca, 'FontName','Times New Roman', 'LineWidth',1.0, 'Box','on');
xlim([0 1]); ylim([0 1]); grid off;
tightfig(fig_b);
export_fig(fig_b, 'Fig1b_condmean');

% ==============================================================
% Figure (c): RCR Profiles
% ==============================================================
fig_c = figure('Color','w', 'Position',[100 100 420 340]);
yyaxis left
plot(X_norm, RCR_y, '.', 'MarkerSize', 8, 'Color', colors.blue); hold on;
plot(XS, movmean(RCR_y_sorted, win), 'Color', colors.blue, 'LineWidth', 2.2);
ylabel('RCR_{Y|X}', 'Color', colors.blue); set(gca, 'YColor', colors.blue);

yyaxis right
plot(Y_norm, RCR_x, '.', 'MarkerSize', 8, 'Color', colors.red);
plot(YS, movmean(RCR_x_sorted, win), 'Color', colors.red, 'LineWidth', 2.2);
ylabel('RCR_{X|Y}', 'Color', colors.red); set(gca, 'YColor', colors.red);

xlabel('Normalized X or Y');
title('RCR profiles', 'FontWeight','bold', 'FontSize',12);
h1 = plot(nan,nan,'.k','MarkerSize',8); h2 = plot(nan,nan,'-k','LineWidth',2.2);
legend([h1 h2], {'Pointwise', 'Smoothed'}, 'Location','southoutside', ...
       'Orientation','horizontal', 'Box','on', 'FontSize',10, 'NumColumns',2);
set(gca, 'FontName','Times New Roman', 'LineWidth',1.0, 'Box','on');
xlim([0 1]); grid off;
tightfig(fig_c);
export_fig(fig_c, 'Fig1c_rcr');

% ==============================================================
% 控制台输出
% ==============================================================
fprintf('================================================================\n');
fprintf(' RCR Results:\n');
fprintf('   VRCR_x = %.4f\n', VRCR_x);
fprintf('   VRCR_y = %.4f\n', VRCR_y);
fprintf('   delta = %.4f\n', delta);
fprintf('   Direction: %s\n', ifelse(ind==1, 'X → Y', 'Y → X'));
fprintf(' Figures saved as:\n');
fprintf('   Fig1a_scatter.[pdf/png/svg]\n');
fprintf('   Fig1b_condmean.[pdf/png/svg]\n');
fprintf('   Fig1c_rcr.[pdf/png/svg]\n');
fprintf(' No grid lines | Times New Roman | SVG included\n');
fprintf(' Ready for Pattern Recognition submission!\n');
fprintf('================================================================\n');

function out = ifelse(cond, a, b)
    if cond, out = a; else, out = b; end
end


function [ind,delta,VRCR_x,VRCR_y,RCR_x,RCR_y,...
          E_X_given_Y,E_Y_given_X,grid_Y,grid_X,...
          X_norm,Y_norm] = RCR_direction_continuous(X,Y)
% RCR_DIRECTION_CONTINUOUS
% Empirical-conditioning version of RCR for continuous data.
%
% This function estimates conditional expectations E[X|Y] and E[Y|X]
% using equal-frequency binning (a nonparametric "empirical conditioning"),
% then computes sample-level RCR terms and their variances VRCR_x, VRCR_y.
% The direction index 'ind' is decided exactly as in the original method:
%   ind = 1  => X -> Y
%   ind = 0  => Y -> X
%
% Inputs:
%   X, Y : column vectors of the same length (n x 1)
%
% Outputs:
%   ind        : binary direction decision (1 means X->Y)
%   delta     : mean of (VRCR_x/VRCR_Y)^2, the decision statistic
%   VRCR_x     : var(RCR_x), i.e. variance of X-vs-Y residual ratio
%   VRCR_y     : var(RCR_y)
%   RCR_x      : per-sample RCR term based on E[X|Y]
%   RCR_y      : per-sample RCR term based on E[Y|X]
%   E_X_given_Y: estimated E[X|Y-bin] on each Y bin (for plotting)
%   E_Y_given_X: estimated E[Y|X-bin] on each X bin (for plotting)
%   grid_Y     : representative Y values (bin centers)
%   grid_X     : representative X values (bin centers)
%   X_norm     : X after min-max normalization to [0,1]
%   Y_norm     : Y after min-max normalization to [0,1]
%
% Empirical conditioning strategy:
%   1. Normalize X,Y to [0,1] (consistent with the manuscript).
%   2. Partition the conditioning variable (e.g. Y_norm) into equal-frequency
%      bins (default nbins = 20). Each bin groups multiple nearby samples.
%   3. For each bin, compute the average of the response variable
%      (e.g. mean(X_norm | Y in that bin)).
%   4. Assign each sample the bin-average as its conditional expectation
%      estimate.
%
%   This avoids the degeneracy that arises in continuous data where
%   "unique(Y)" would otherwise assign each sample to a singleton group.
%
%   The rest of the pipeline (RCR_x, RCR_y, VRCR_x, VRCR_y, and direction
%   decision by comparing VRCR_x vs VRCR_y) is unchanged.

% ---------------- parameters ----------------
nbins   = 20;      % number of equal-frequency bins for empirical conditioning
epsilon = 1e-8;

% ---------------- sanity checks ----------------
X = X(:);
Y = Y(:);
n = length(X);
if length(Y) ~= n
    error('X and Y must have the same length.');
end

% ---------------- normalization to [0,1] ----------------
X_norm = (X - min(X)) / (max(X) - min(X) + eps);
Y_norm = (Y - min(Y)) / (max(Y) - min(Y) + eps);

% =====================================================================
% Estimate E[X|Y] using equal-frequency binning on Y_norm
% =====================================================================
qY = linspace(0,1,nbins+1);
edgesY = quantile(Y_norm,qY);
edgesY = unique(edgesY);          % remove duplicate edges (handle flat tails)

% assign each sample to a Y-bin
binYidx = discretize(Y_norm, edgesY);

nBinsY = max(binYidx);
E_X_given_Y_bins = nan(nBinsY,1);
bin_centers_Y    = nan(nBinsY,1);

for b = 1:nBinsY
    idxb = (binYidx == b);
    if any(idxb)
        E_X_given_Y_bins(b) = mean(X_norm(idxb));
        bin_centers_Y(b)    = mean(Y_norm(idxb));
    else
        E_X_given_Y_bins(b) = NaN;
        bin_centers_Y(b)    = NaN;
    end
end

% map each sample to its bin-average E[X|Y]
EX_given_Y_each = E_X_given_Y_bins(binYidx);
% fill any NaN with grand mean just in case
nanmask = isnan(EX_given_Y_each);
if any(nanmask)
    EX_given_Y_each(nanmask) = nanmean(EX_given_Y_each);
end

% =====================================================================
% Estimate E[Y|X] using equal-frequency binning on X_norm
% =====================================================================
qX = linspace(0,1,nbins+1);
edgesX = quantile(X_norm,qX);
edgesX = unique(edgesX);

binXidx = discretize(X_norm, edgesX);

nBinsX = max(binXidx);
E_Y_given_X_bins = nan(nBinsX,1);
bin_centers_X    = nan(nBinsX,1);

for b = 1:nBinsX
    idxb = (binXidx == b);
    if any(idxb)
        E_Y_given_X_bins(b) = mean(Y_norm(idxb));
        bin_centers_X(b)    = mean(X_norm(idxb));
    else
        E_Y_given_X_bins(b) = NaN;
        bin_centers_X(b)    = NaN;
    end
end

EY_given_X_each = E_Y_given_X_bins(binXidx);
nanmask2 = isnan(EY_given_X_each);
if any(nanmask2)
    EY_given_X_each(nanmask2) = nanmean(EY_given_X_each);
end

% =====================================================================
% Compute sample-level RCR terms (keep your original ratio form)
% =====================================================================
sigma_X = std(X_norm); if sigma_X < 1e-6, sigma_X = 1; end
sigma_Y = std(Y_norm); if sigma_Y < 1e-6, sigma_Y = 1; end

RCR_x = (X_norm - EX_given_Y_each) / sigma_X;
RCR_y = (Y_norm - EY_given_X_each) / sigma_Y;

% global variances
VRCR_x = var(RCR_x);
VRCR_y = var(RCR_y);
delta = mean((VRCR_x/VRCR_y).^2)
% direction decision rule
if delta>=1
    ind=1
else
    ind=0
end
% =====================================================================
% expose grids and conditional curves for plotting
% =====================================================================
% For interpretability in plots:
%   grid_Y vs E_X_given_Y : "estimated E[X|Y]"
%   grid_X vs E_Y_given_X : "estimated E[Y|X]"
E_X_given_Y = E_X_given_Y_bins;
E_Y_given_X = E_Y_given_X_bins;
grid_Y      = bin_centers_Y;
grid_X      = bin_centers_X;
end

% ==============================================================
% ==============================================================
% 内置 tightfig 和 export_fig（支持 SVG）
% ==============================================================
function tightfig(fig_h)
    ax = get(fig_h, 'CurrentAxes');
    if ~isempty(ax)
        ti = get(ax, 'TightInset');
        pos = [ti(1), ti(2), 1-ti(1)-ti(3), 1-ti(2)-ti(4)];
        set(ax, 'Position', pos);
    end
    set(fig_h, 'PaperPositionMode', 'auto');
end

function export_fig(fig_h, name)
    % PDF (vector, LaTeX)
    print(fig_h, [name '.pdf'], '-dpdf', '-bestfit');
    % PNG (300 DPI)
    print(fig_h, [name '.png'], '-dpng', '-r300');
    % SVG (web/editable)
    print(fig_h, [name '.svg'], '-dsvg');
end