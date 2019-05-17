import pandas as pd
import gpflow
import numpy as np

import matplotlib.pyplot as plt

country = 'sierraleone'
df = pd.read_csv('data/previous-case-counts-%s.csv' % country)
df['WHO report date'] = pd.to_datetime(df['WHO report date'], format="%d/%m/%Y")
df['delta_time_days'] = (df['WHO report date'] - df['WHO report date'].min()).dt.days
df = df.sort_values('delta_time_days')
print(df)
X = df['delta_time_days'].values[:, np.newaxis].astype('float')
Y = df['Total Cases'].values[:, np.newaxis].astype('float')

with gpflow.defer_build():
    k = gpflow.kernels.Matern52(1)
    m = gpflow.models.GPR(X, Y, kern=k)
    m.likelihood.variance.trainable = False
    m.likelihood.variance = 300**2
m.compile()
gpflow.train.ScipyOptimizer().minimize(m)
print(m.as_pandas_table())

xx = np.linspace(np.min(X), np.max(X), 200).reshape(200, 1)
mean, var = m.predict_y(xx)
samples = m.predict_f_samples(xx, 10)
print(np.sum(m.predict_density(X, Y)))
print(np.sum(m.predict_density(X, Y+100)))

plt.figure(figsize=(12, 6))
plt.plot(X, Y, 'kx', mew=2)
plt.plot(xx, mean, 'C0', lw=2)
for i in range(10):
    plt.plot(xx[:, 0], samples[i, :, 0])
plt.fill_between(xx[:, 0], mean[:, 0] - np.sqrt(var[:, 0]), mean[:, 0] + np.sqrt(var[:, 0]), color='C0', alpha=0.2)
plt.show()
