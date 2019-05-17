from __future__ import  division, print_function

import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import gpflow
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
        with gpflow.defer_build():
            k = gpflow.kernels.Matern52(1)
            self.mc = gpflow.models.GPR(df['delta_time_days'].values[:, np.newaxis].astype('float'),
                                        df['Total Cases'].values[:, np.newaxis].astype('float'), kern=k)
            self.mc.likelihood.variance.trainable = False
            self.mc.likelihood.variance = 300 ** 2
        self.mc.compile()
        gpflow.train.ScipyOptimizer().minimize(self.mc)
        print(self.mc.as_pandas_table())
        with gpflow.defer_build():
            k = gpflow.kernels.Matern52(1)
            self.md = gpflow.models.GPR(df['delta_time_days'].values[:, np.newaxis].astype('float'),
                                        df['Total Deaths'].values[:, np.newaxis].astype('float'), kern=k)
            self.md.likelihood.variance.trainable = False
            self.md.likelihood.variance = 300 ** 2
        self.md.compile()
        gpflow.train.ScipyOptimizer().minimize(self.md)
        print(self.md.as_pandas_table())

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
        loglike = self.mc.predict_density(self.df['delta_time_days'][:, np.newaxis].astype('float'), sol[1:, 4:5]) + \
                  self.md.predict_density(self.df['delta_time_days'][:, np.newaxis].astype('float'), sol[1:, 5:6])
        return np.sum(loglike)


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

    log_dir = os.path.join(args.log_dir, args.country)
    sampler = NestedSampler(args.x_dim, loglike, transform=transform, log_dir=log_dir, num_live_points=args.num_live_points,
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
    parser.add_argument("--num_live_points", type=int, default=100)
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
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--country', type=str, default='guinea')
    parser.add_argument('--N', type=int, default=1000000)

    args = parser.parse_args()
    main(args)
