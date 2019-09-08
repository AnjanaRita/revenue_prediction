import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-3.998307350385423
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        make_union(
            StackingEstimator(estimator=make_pipeline(
                StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.5, min_samples_leaf=7, min_samples_split=11, n_estimators=100)),
                StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.45, min_samples_leaf=9, min_samples_split=3, n_estimators=100)),
                RandomForestRegressor(bootstrap=True, max_features=0.4, min_samples_leaf=17, min_samples_split=10, n_estimators=100)
            )),
            make_union(
                FunctionTransformer(copy),
                FunctionTransformer(copy)
            )
        )
    ),
    RandomForestRegressor(bootstrap=False, max_features=0.8500000000000001, min_samples_leaf=5, min_samples_split=2, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
