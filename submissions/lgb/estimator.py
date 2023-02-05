from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator



def compute_rolling_std(X_df, feature, sub_feature, 
        time_window , l_alpha, types = ["rolling", "emv"], center=False):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (time_window) of a defined feature

    Parameters
    ----------
    X : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling std from
    time_window : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """

    X_df = X_df.copy()[sub_feature]

    for type in types :
        if type == "rolling":
            for time in time_window :
                    print(type + "here")
                    #data = X_df[feature].rolling(time, center=center).agg(("sum",  "min", "max","kurt", "skew", "median", lambda x: x.quantile(0.75)))
                    data = X_df[feature].rolling(time, center=center).agg(("mean", "std", "median","min", "max","kurt", "skew", "sum"))
                    name = [f"_{str(time)}_".join(a) for a in data.columns.to_list()]
                    X_df[name] = data.values
                    X_df[name] = X_df[name].ffill().bfill()
                    X_df[name] = X_df[name].astype(float)
        if type == "emv":

            for alpha in l_alpha:
                    print(type + "here")
                    data = X_df[feature].ewm(alpha=alpha).agg({"mean", "std"})
                    name = [f"_{str(alpha)}_".join(a) for a in data.columns.to_list()]
                    X_df[name] = data.values
                    X_df[name] = X_df[name].ffill().bfill()
                    X_df[name] = X_df[name].astype(float)
    return X_df


class FeatureExtractor(BaseEstimator):
    def __init__(self, feature , window= ["2h", "24h"], alpha = [0.2, 0.5, 0.7]) -> None:
        super().__init__()
        self.window = window
        self.feature = feature
        self.alpha = alpha
    def fit(self, X, y):
        return self

    def transform(self, X):
        return compute_rolling_std(X_df=X, feature=self.feature, sub_feature=self.feature, time_window=self.window,
        l_alpha=self.alpha)


def get_estimator():

    feature_extractor = FeatureExtractor(feature=['B', 'Range F 1', 'Range F 13', 'Range F 9', 'Vth', 'Vx', 'Vy',
    'Beta', 'Pdyn', 'RmsBob'])

    params={'boosting_type': 'gbdt', 'learning_rate': 0.047403318065537894, 'reg_alpha': 61.819153561585296, 'reg_lambda': 1.637753769029222}

    classifier = LGBMClassifier(**params, n_jobs=-1)
    pipe = make_pipeline(feature_extractor, classifier)
    return pipe
