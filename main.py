import csv
import pandas as pd
import itertools as it
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from Fortuna import RandomValue, front_linear, back_linear
from Fortuna import front_gauss, back_gauss, front_poisson, back_poisson


random_method = RandomValue((
    ('Front Linear', front_linear),
    ('Back Linear', back_linear),
    ('Front Gauss', front_gauss),
    ('Back Gauss', back_gauss),
    ('Front Poisson', front_poisson),
    ('Back Poisson', back_poisson),
))


def make_csv(name, var, n_rows, n_cols):
    with open(name, 'w', newline='') as csv_file:
        spam = csv.writer(csv_file, delimiter=',')
        spam.writerow(it.chain(
            ('Method', ),
            (f'Value {i+1}' for i in range(n_cols))),
        )
        for i in range(n_rows):
            name, method = random_method()
            spam.writerow(it.chain(
                (name, ),
                (method(var) + 1 for _ in range(n_cols))),
            )


dice = (4, 6, 8, 10, 12, 20)
for d in dice:
    rows = 10000
    cols = 10
    make_csv(f'data/method_{d}.csv', d, rows, cols)

data = {
    n: pd.read_csv(f'data/method_{n}.csv') for n in dice
}

print("\nValidation Accuracy:")
for N in data.keys():
    X_train, X_val = train_test_split(data[N], random_state=42)
    y_train = X_train['Method']
    X_train = X_train.drop(columns=['Method'])
    y_val = X_val['Method']
    X_val = X_val.drop(columns=['Method'])
    model = RandomForestClassifier(
        bootstrap=False,
        criterion='gini',
        max_depth=12,
        max_features=1,
        n_estimators=128,
        n_jobs=-1,
        random_state=42,
        warm_start=True,
    )
    model.fit(X_train, y_train)
    print(f"\td{N}: \t{100 * model.score(X_val, y_val):.2f}%")
