import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-6317138773280758.0
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    Normalizer(norm="l1"),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.95, learning_rate=0.1, loss="huber", max_depth=2, max_features=0.7000000000000001, min_samples_leaf=5, min_samples_split=17, n_estimators=100, subsample=0.9500000000000001)),
    ExtraTreesRegressor(bootstrap=True, max_features=0.6500000000000001, min_samples_leaf=2, min_samples_split=9, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)