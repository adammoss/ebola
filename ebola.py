from __future__ import  division, print_function

import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.stats import multivariate_normal
from nnest.nested import NestedSampler


class Ebola(object):

    def __init__(self, N, country, plot=False):
        df = pd.read_csv('data/previous-case-counts-%s.csv' % country)
        df['WHO report date'] = pd.to_datetime(df['WHO report date'], format="%d/%m/%Y")
        df['delta_time_days'] = (df['WHO report date'] - df['WHO report date'].min()).dt.days
        df = df.sort_values('delta_time_days')
        print(df)
        self.df = df
        self.N = N
        self.country = country
        self.plot = plot
        # Differential case counts
        self.delta_cases = df['Total Cases'].values[1:] - df['Total Cases'].values[:-1]
        # Differential death counts
        self.delta_deaths = df['Total Deaths'].values[1:] - df['Total Deaths'].values[:-1]
        # GP fit
        kernel = 1 * RBF(length_scale=1000, length_scale_bounds=(10, 10000)) + \
                 WhiteKernel(noise_level=1000, noise_level_bounds=(10, 1000))
        gp = GaussianProcessRegressor(kernel=kernel).fit(df['delta_time_days'].values[:, np.newaxis],
                                                         df['Total Cases'].values)
        self.cases_mean, self.cases_cov = gp.predict(df['delta_time_days'].values[:, np.newaxis], return_cov=True)
        gp = GaussianProcessRegressor(kernel=kernel).fit(df['delta_time_days'].values[:, np.newaxis],
                                                         df['Total Deaths'].values)
        self.deaths_mean, self.deaths_cov = gp.predict(df['delta_time_days'].values[:, np.newaxis], return_cov=True)

        if self.plot:
            X_ = df['delta_time_days'].values
            plt.plot(X_, self.cases_mean, 'k', lw=3, zorder=9)
            plt.fill_between(X_, self.cases_mean - np.sqrt(np.diag(self.cases_cov)), self.cases_mean + np.sqrt(np.diag(self.cases_cov)), alpha=0.5, color='k')
            plt.scatter(df['delta_time_days'].values, df['Total Cases'].values, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
            plt.show()

            plt.plot(X_, self.deaths_mean, 'k', lw=3, zorder=9)
            plt.fill_between(X_, self.deaths_mean - np.sqrt(np.diag(self.deaths_cov)), self.deaths_mean + np.sqrt(np.diag(self.deaths_cov)), alpha=0.5, color='k')
            plt.scatter(df['delta_time_days'].values, df['Total Deaths'].values, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
            plt.show()

    def rate(self, y, t, beta, k, tau, sigma, gamma, f):
        S, E, I, R, C, D = y
        beta_t = beta * np.exp(-k * (t - tau))
        dydt = [
            -beta_t * S * I / self.N,
            beta_t * S * I / self.N - sigma * E,
            sigma * E - gamma * I,
            (1 - f) * gamma * I,
            sigma * E,
            f * gamma * I
        ]
        return dydt

    def solve(self, beta, k, tau, sigma, gamma, f, offset):
        y0 = [self.N - 1, 0, 1, 0, 1, 0]
        # Offset initial time by constant
        t = self.df['delta_time_days'].values + offset
        t[t < 0] = 0
        t = np.insert(t, 0, 0, axis=0)
        sol = odeint(self.rate, y0, t, args=(beta, k, tau, sigma, gamma, f))
        if self.plot:
            f, ax = plt.subplots()
            ax.set_title(self.country)
            ax.plot(self.df['delta_time_days'], sol[1:, 4], linestyle='solid', marker='None', color='red')
            ax.plot(self.df['delta_time_days'], self.df['Total Cases'], color='red', mfc='None', marker='o', linestyle='None')
            ax.plot(self.df['delta_time_days'], sol[1:, 5], linestyle='solid', marker='None', color='blue')
            ax.plot(self.df['delta_time_days'], self.df['Total Deaths'], color='blue', mfc='None', marker='o', linestyle='None')
            plt.show()
        return sol

    def __call__(self, theta):
        sol = self.solve(*theta)
        loglike = multivariate_normal.logpdf(sol[1:, 4], mean=self.cases_mean, cov=self.cases_cov) + \
                  multivariate_normal.logpdf(sol[1:, 5], mean=self.deaths_mean, cov=self.deaths_cov)
        return loglike


def main(args):

    e = Ebola(args.N, args.country, plot=False)

    def loglike(z):
        return np.array([e(x) for x in z])

    def transform(x):
        return np.array([
            0.3 + 0.25 * x[:, 0],
            0.001 + 0.001 * x[:, 1],
            5 + 5 * x[:, 2],
            0.15 + 0.1 * x[:, 3],
            0.15 + 0.1 * x[:, 4],
            0.5 + 0.4 * x[:, 5],
            100 + 100 * x[:, 6]
        ]).T

    sampler = NestedSampler(args.x_dim, loglike, transform=transform, log_dir=args.log_dir, num_live_points=args.num_live_points,
                            hidden_dim=args.hidden_dim, num_layers=args.num_layers, num_blocks=args.num_blocks, num_slow=args.num_slow,
                            use_gpu=args.use_gpu)
    sampler.run(train_iters=args.train_iters, mcmc_steps=args.mcmc_steps, volume_switch=args.switch, noise=args.noise)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=7,
                        help="Dimensionality")
    parser.add_argument('--train_iters', type=int, default=50,
                        help="number of train iters")
    parser.add_argument("--mcmc_steps", type=int, default=0)
    parser.add_argument("--num_live_points", type=int, default=1000)
    parser.add_argument('--switch', type=float, default=-1)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('-use_gpu', action='store_true')
    parser.add_argument('--flow', type=str, default='nvp')
    parser.add_argument('--num_blocks', type=int, default=5)
    parser.add_argument('--noise', type=float, default=-1)
    parser.add_argument("--test_samples", type=int, default=0)
    parser.add_argument('--run_num', type=str, default='')
    parser.add_argument('--num_slow', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='logs/mog4')
    parser.add_argument('--country', type=str, default='guinea')
    parser.add_argument('--N', type=int, default=1000000)

    args = parser.parse_args()
    main(args)
