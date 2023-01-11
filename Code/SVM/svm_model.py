import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import make_scorer
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from imblearn.metrics import specificity_score
from contextlib import contextmanager
from timeit import default_timer


SEED = 42


def prepare_data(df, split=True):
    X, y = df.drop('depression.symptoms', axis=True), df['depression.symptoms']

    scaler = MinMaxScaler()
    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=SEED)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test
    else:
        X = scaler.fit_transform(X)
        return X, y


def train(X_train, y_train):
    # apply Bayesian search to find optimal hyperparameter set
    opt = BayesSearchCV(
         SVC(class_weight='balanced'),
         {
             'C': Real(1e-6, 1e-1, prior='log-uniform'),
             'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
             'degree': Integer(1, 4),
             'kernel': Categorical(['linear', 'poly', 'rbf']),
         },
         n_iter=25,
         cv=5,
         return_train_score=True,
         n_jobs=3,
         verbose=0,
         random_state=SEED
    )

    opt.fit(X_train, y_train)
    return opt


def calulate_score(opt, X_test, y_test):
    # precision, recall, f1
    scorings = {
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted',
    }
    scores = cross_validate(
        opt.best_estimator_,
        X_test, y_test,
        cv=10,
        scoring=scorings,
        n_jobs=3
    )

    # specificity
    scorer = make_scorer(specificity_score, average='weighted')
    spec_score = cross_val_score(
        opt.best_estimator_,
        X_test, y_test,
        cv=10,
        scoring=scorer,
        n_jobs=3
    )
    scores['test_specificity'] = spec_score

    # clean scores
    scores.pop('fit_time')
    scores.pop('score_time')

    summary_stats = dict(map(lambda kv: (kv[0], {'mean': np.mean(kv[1]), 'std': np.std(kv[1])}), scores.items()))

    return summary_stats


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start


df = pd.read_pickle('./svm_df.pkl')
with elapsed_timer() as elapsed:
    X, y = prepare_data(df, split=False)
    print("Data acquired at %.2f seconds. Starting the training" % elapsed())
    opt = train(X, y)
    print("Training complete at %.2f seconds. Starting evaluation" % elapsed())
    scores = calulate_score(opt, X, y)
    print("Evaluation complete at %.2f seconds." % elapsed())

print(scores)
