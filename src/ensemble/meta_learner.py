import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from scipy.special import expit, logit

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

class LogOddsStacker:
    def __init__(self, seed):
        self.seed = seed
        self.base_classifiers = self._init_base_classifiers()
        self.meta_learner = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced')
        
    def _init_base_classifiers(self):
        clfs = [
            ('rf', RandomForestClassifier(n_estimators=450, max_depth=17, class_weight='balanced', random_state=self.seed)),
            ('gbc', GradientBoostingClassifier(n_estimators=250, learning_rate=0.02, max_depth=6, random_state=self.seed)),
            ('lr', LogisticRegression(C=0.25, class_weight='balanced', random_state=self.seed, max_iter=3000)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(224, 112), activation='relu', random_state=self.seed, early_stopping=True)),
            ('svm', SVC(C=2.5, kernel='rbf', class_weight='balanced', probability=True, random_state=self.seed))
        ]
        if HAS_XGB:
            clfs.append(('xgb', xgb.XGBClassifier(n_estimators=400, learning_rate=0.008, random_state=self.seed, eval_metric='auc')))
        if HAS_LGB:
            clfs.append(('lgb', lgb.LGBMClassifier(n_estimators=400, learning_rate=0.008, class_weight='balanced', random_state=self.seed, verbose=-1)))
        return clfs

    def fit(self, X, y):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        oof_probs = np.zeros((X.shape[0], len(self.base_classifiers)))
        
        for i, (name, clf) in enumerate(self.base_classifiers):
            for train_idx, val_idx in skf.split(X, y):
                clf.fit(X[train_idx], y[train_idx])
                probs = clf.predict_proba(X[val_idx])[:, 1]
                oof_probs[val_idx, i] = probs
            
            clf.fit(X, y)
            
        oof_log_odds = logit(np.clip(oof_probs, 1e-6, 1-1e-6))
        self.meta_learner.fit(oof_log_odds, y)
        return self

    def predict_proba(self, X):
        base_probs = np.zeros((X.shape[0], len(self.base_classifiers)))
        for i, (name, clf) in enumerate(self.base_classifiers):
            base_probs[:, i] = clf.predict_proba(X)[:, 1]
            
        base_log_odds = logit(np.clip(base_probs, 1e-6, 1-1e-6))
        return self.meta_learner.predict_proba(base_log_odds)

def get_ensemble(seed):
    return LogOddsStacker(seed)

def calibrate_ensemble(ensemble, X_train, y_train):
    ensemble.fit(X_train, y_train)
    return ensemble, 1.0

def temperature_scale(probs, temperature):
    return probs
