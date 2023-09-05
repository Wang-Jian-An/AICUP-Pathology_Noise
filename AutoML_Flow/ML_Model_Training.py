import os, gzip, pickle, itertools, optuna
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.feature_selection import RFECV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from mlxtend.feature_selection import *
from .regression_model_evaluation import model_evaluation as regression_model_evaluation
from .two_class_model_evaluation import model_evaluation as two_class_model_evaluation

"""
關鍵：用訓練資料訓練模型、用驗證資料確認超參數調整、用測試資料實施最後的模型評估

trainData, valiData, inputFeatures, target, HTMetric, target_type

"""


class model_training_and_hyperparameter_tuning:
    def __init__(
        self,
        trainData: pd.DataFrame,
        valiData: pd.DataFrame,
        inputFeatures: list,
        target,
        target_type,
        model_name,
        feature_selection_method,
        HTMetric,
        hyperparameter_tuning_method,
        thresholdMetric: str = None
    ):
        """
        trainData：訓練資料。type: pd.DataFrame
        valiData：驗證資料，用於超參數調整。type: pd.DataFrame
        inputFeature：輸入特徵。
        feature_selection_method: SBS、SFS、SFBS、SFFS、RFECV
        
        """

        self.trainData = trainData
        self.valiData = valiData
        self.trainData_valiData = pd.concat([self.trainData, self.valiData], axis=0)
        self.inputFeatures = inputFeatures
        self.target = target
        self.target_type = target_type
        self.model_name = model_name
        self.n_trials = 3 if "Extra Tree" in self.model_name else 2
        self.HTMetric = HTMetric
        self.thresholdMetric = thresholdMetric
        self.feature_selection_method = feature_selection_method
        self.hyperparameter_tuning_method = hyperparameter_tuning_method
        if self.target_type == "classification" and self.trainData[self.target].unique().shape[0] == 2:
            self.define_best_thres = True
        else:
            self.define_best_thres = False
        return

    def model_training(self):
        if self.feature_selection_method != "None":
            self.feature_selection()

        if self.hyperparameter_tuning_method == "TPESampler":
            ### Use Optuna to tune hyperparameter ###
            study = optuna.create_study(direction="minimize")
            study.optimize(self.objective_function, n_trials=self.n_trials, n_jobs = 1)
            ### Use Optuna to tune hyperparameter ###

            ### Output the result of hyperparameter tuning ###
            study_trial_data = study.trials_dataframe()
            study_trial_data["Model"] = self.model_name

            try:
                fig = optuna.visualization.plot_param_importances(study)
            except:
                fig = None
            ### Output the result of hyperparameter tuning ###

            bestHyperParams = study.best_params
#             self.model = self.choose_one_model()
#             self.model.set_params(**study.best_params)
        else:
            bestHyperParams = dict()
#             self.model = self.choose_one_model()
        
        # Define best threshold for binary classification
        if self.define_best_thres:
            model = self.choose_one_model(params = bestHyperParams)
            model.fit(self.trainData[self.inputFeatures], self.trainData[self.target])
            vali_yhat = model.predict_proba(self.valiData[self.inputFeatures])
            best_thres_optimizer = minimize_scalar(
                self.find_best_thres, 
                args = (vali_yhat, self.valiData[self.target], ),
                bounds=(0.01, 0.99)
            )
            best_thres = best_thres_optimizer.x
            print("最佳 Threshold", best_thres)
        else:
            best_thres = None
        self.model = self.choose_one_model(params = bestHyperParams)
        self.model.fit(
            self.trainData_valiData[self.inputFeatures],
            self.trainData_valiData[self.target],
        )
        return {
            "Features": self.inputFeatures,
            "Model": self.model,
            "Best_Thres": best_thres, 
            "Hyperparameter_Tuning": study_trial_data if self.hyperparameter_tuning_method == "TPESampler" else None, 
            "Param_Importance": fig if self.hyperparameter_tuning_method == "TPESampler" else None,
        }

    def find_best_thres(self, x, vali_yhat_proba, vali_target):
        vali_yhat = np.where(vali_yhat_proba[:, -1] > x, 1, 0)
        evaluation_result = two_class_model_evaluation(
            ytrue = vali_target,
            ypred = vali_yhat,
            ypred_proba = vali_yhat_proba
        )
        score = -1 * evaluation_result[self.thresholdMetric]
        return score
    
    def feature_selection(self):
        if self.feature_selection_method == "SFS":
            featureSelectionObj = SequentialFeatureSelector(
                estimator=self.choose_one_model(),
                k_features=len(self.inputFeatures) // 2,
                forward=True,
                floating=False,
                scoring= "neg_log_loss" if self.HTMetric == "cross_entropy" else self.HTMetric,
                verbose=1,
                n_jobs=-1,
                cv=5,
            ).fit(
                X=self.trainData_valiData[self.inputFeatures],
                y=self.trainData_valiData[self.target],
            )

        elif self.feature_selection_method == "SBS":
            featureSelectionObj = SequentialFeatureSelector(
                estimator=self.choose_one_model(),
                k_features=len(self.inputFeatures) // 2,
                forward=False,
                floating=False,
                scoring="neg_log_loss" if self.HTMetric == "cross_entropy" else self.HTMetric,
                verbose=1,
                n_jobs=-1,
                cv=5,
            ).fit(
                X=self.trainData_valiData[self.inputFeatures],
                y=self.trainData_valiData[self.target],
            )
        elif self.feature_selection_method == "SFFS":
            featureSelectionObj = SequentialFeatureSelector(
                estimator=self.choose_one_model(),
                k_features=len(self.inputFeatures) // 2,
                forward=True,
                floating=True,
                scoring="neg_log_loss" if self.HTMetric == "cross_entropy" else self.HTMetric,
                verbose=1,
                n_jobs=-1,
                cv=5,
            ).fit(
                X=self.trainData_valiData[self.inputFeatures],
                y=self.trainData_valiData[self.target],
            )
        elif self.feature_selection_method == "SFBS":
            featureSelectionObj = SequentialFeatureSelector(
                estimator=self.choose_one_model(),
                k_features=len(self.inputFeatures) // 2,
                forward=True,
                floating=True,
                scoring="neg_log_loss" if self.HTMetric == "cross_entropy" else self.HTMetric,
                verbose=1,
                n_jobs=-1,
                cv=5,
            ).fit(
                X=self.trainData_valiData[self.inputFeatures],
                y=self.trainData_valiData[self.target],
            )
        elif self.feature_selection_method == "RFECV":
            featureSelectionObj = RFECV(
                estimator=self.choose_one_model(),
                min_features_to_select=len(self.inputFeatures) // 2,
                verbose=1,
                n_jobs=-1,
                cv=5,
            ).fit(
                X=self.trainData_valiData[self.inputFeatures],
                y=self.trainData_valiData[self.target],
            )

        if self.feature_selection_method == "RFECV":
            self.inputFeatures = featureSelectionObj.feature_names_in_.tolist()
        else:
            self.inputFeatures = list(featureSelectionObj.k_feature_names_)

    def choose_one_model(self, params = dict()):
        if self.target_type == "classification":
            if self.model_name == "Random Forest with Entropy":
                self.model = RandomForestClassifier(
                    **{"criterion": "entropy", "n_jobs": -1, **params}
                )
            elif self.model_name == "Random Forest with Gini":
                self.model = RandomForestClassifier(
                    **{"criterion": "gini", "n_jobs": -1, **params}
                )
            elif self.model_name == "ExtraTree with Entropy":
                self.model = ExtraTreeClassifier(**{"criterion": "entropy", **params})
            elif self.model_name == "ExtraTree with Gini":
                self.model = ExtraTreeClassifier(**{"criterion": "gini", **params})
            elif self.model_name == "XGBoost":
                self.model = XGBClassifier(**params)
            elif self.model_name == "CatBoost":
                self.model = CatBoostClassifier(**params)
            elif self.model_name == "LightGBM":
                self.model = LGBMClassifier(**params)
            elif self.model_name == "LightGBM with ExtraTrees":
                self.model = LGBMClassifier(
                    **{"extra_trees": True, "min_data_in_leaf": 20},
                    **params
                )
            elif self.model_name == "NeuralNetwork":
                pass
            pass
        elif self.target_type == "regression":
            if self.model_name == "Random Forest with squared_error":
                self.model = RandomForestRegressor(**{"criterion": "squared_error", **params})
            elif self.model_name == "Random Forest with absolute_error":
                self.model = RandomForestRegressor(**{"criterion": "absolute_error", **params})
            elif self.model_name == "Random Forest with friedman_mse":
                self.model = RandomForestRegressor(**{"criterion": "friedman_mse", **params})
            elif self.model_name == "ExtraTree with squared_error":
                self.model = ExtraTreeRegressor(**{"criterion": "squared_error", **params})
            elif self.model_name == "ExtraTree with absolute_error":
                self.model = ExtraTreeRegressor(**{"criterion": "absolute_error", **params})
            elif self.model_name == "ExtraTree with friedman_mse":
                self.model = ExtraTreeRegressor(**{"criterion": "friedman_mse", **params})
            elif self.model_name == "XGBoost":
                self.model = XGBRegressor(**params)
            elif self.model_name == "CatBoost":
                self.model = CatBoostRegressor(**params)
            elif self.model_name == "LightGBM":
                self.model = LGBMRegressor(**params)
            elif self.model_name == "LightGBM with ExtraTrees":
                self.model = LGBMRegressor(
                    **{"extra_trees": True, "min_data_in_leaf": 20},
                    **params
                )
        return self.model

    def objective_function(self, trial):
        oneModel = self.choose_one_model()
        oneModel.set_params(**self.model_parameter_for_optuna(trial))
        oneModel.fit(self.trainData[self.inputFeatures], self.trainData[self.target])

        # 根據二分類任務、多分類任務或是迴歸任務，給予不同評估指標的設定。
        if self.target_type == "classification" and self.trainData[self.target].unique().__len__() == 2:
            allMetric = two_class_model_evaluation(
                ytrue = self.valiData[self.target],
                ypred = oneModel.predict(self.valiData[self.inputFeatures]),
                ypred_proba = oneModel.predict_proba(self.valiData[self.inputFeatures])
            )
            metric = allMetric[self.HTMetric]
            if self.HTMetric == "cross_entropy":
                return metric
        elif self.target_type == "classification" and self.trainData[self.target].unique().__len__() > 2:
            if self.HTMetric == "accuracy":
                metric = accuracy_score(
                    y_true=self.valiData[self.target],
                    y_pred=oneModel.predict(self.valiData[self.inputFeatures]),
                )
            elif self.HTMetric == "f1":
                metric = f1_score(
                    y_true=self.valiData[self.target],
                    y_pred=oneModel.predict(self.valiData[self.inputFeatures]),
                    average = "macro"
                )
            elif self.HTMetric == "roc_auc":
                metric = roc_auc_score(
                    y_true=self.valiData[self.target],
                    y_pred=oneModel.predict_proba(self.valiData[self.inputFeatures]),
                    average = "macro"
                )   
            elif self.HTMetric == "cross_entropy":
                metric = log_loss(
                    y_true=self.valiData[self.target],
                    y_pred=oneModel.predict_proba(self.valiData[self.inputFeatures])
                )   
        elif self.target_type == "regression":
            allMetric = regression_model_evaluation(
                ytrue = self.valiData[self.target],
                ypred = oneModel.predict(self.valiData[self.inputFeatures])
            )
            metric = allMetric[self.HTMetric]
        return -metric

    def model_parameter_for_optuna(self, trial):
        if "Random Forest" in self.model_name:
            return {
                "n_estimators": trial.suggest_int("n_estimators", 2, 10000),
                "max_depth": trial.suggest_int("max_depth", 5, 500),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2"]
                ),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 10000),
                "oob_score": trial.suggest_categorical("oob_score", [False, True]),
                "n_jobs": -1
            }
        
        elif "LightGBM" in self.model_name:
            return {
                # "boosting_type": trial.suggest_categorical("boosting_type", ['gbdt', "rf"]),
                "num_leaves": trial.suggest_int("num_leaves", 2, 100),
                "max_depth": trial.suggest_int("max_depth", 2, 100),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1),
                "n_estimators": trial.suggest_int("n_estimators", 50, 5000),
                "subsample": trial.suggest_float("subsample", 0.0, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.9),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 0.9),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 0.9),
            }
        
        elif "ExtraTree" in self.model_name:
            return {
                "splitter": trial.suggest_categorical("splitter", ["random", "best"]),
                "max_depth": trial.suggest_int("max_depth", 2, 1000),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2"]
                ),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 2000),
            }
        elif self.model_name == "XGBoost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 2, 10000),
                "max_depth": trial.suggest_int("max_depth", 5, 500),
                "max_leaves": trial.suggest_int("max_leaves", 2, 300),
                "max_bin": trial.suggest_int("max_bin", 2, 100),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2),
                "tree_method": trial.suggest_categorical(
                    "tree_method", ["exact", "approx", "hist"]
                ),
                "subsample": trial.suggest_float("subsample", 0.1, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.9),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 0.9),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 0.9),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 0.9),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 0.9),
            }
        elif self.model_name == "NGBoost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1),
                # "minibatch_frac": trial.suggest_float("minibatch_frac", 0.1, 1.0),
                # "col_sample": trial.suggest_float("col_sample", 0.1, 0.5),
                # "tol": trial.suggest_float("tol", 1e-6, 1e-2),
            }
        elif self.model_name == "CatBoost":
            return {
                "iterations": 500, 
                "od_type": trial.suggest_categorical(
                    "od_type", ["IncToDec", "Iter"]
                ),  # 過擬合偵測器
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 1e-1
                ),  # 學習率
                "depth": trial.suggest_int("depth", 5, 16),  # 樹的深度
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg", 0.01, 0.9
                ),  # L2 Regularization
                "random_strength": trial.suggest_float("random_strength", 0.1, 10),
                "bagging_temperature": trial.suggest_int("bagging_temperature", 1, 100),
                "border_count": trial.suggest_int("border_count", 128, 512),
                "grow_policy": trial.suggest_categorical(
                    "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
                ),  # 定義如何建構 greedy tree
                "verbose": 0
            }

        elif self.model_name == "NeuralNetwork":
            hidden_layer_sizes_trial = [
                k
                for i in range(1, 8)
                for k in [
                    j
                    for j in itertools.combinations(
                        [50, 100, 150, 200, 150, 100, 50], i
                    )
                ]
            ]
            return {
                "hidden_layer_sizes": trial.suggest_categorical(
                    "hidden_layer_sizes", hidden_layer_sizes_trial
                ),
                "activation": "relu",
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-3),
                "learning_rate": "adaptive",
                "learning_rate_init": trial.suggest_float(
                    "learning_rate_init", 1e-5, 1e-1
                ),
                "max_iter": 500,
                "tol": 1e-4,
                "momentum": trial.suggest_float("momentum", 0.01, 0.9),
                "early_stopping": True,
                "n_iter_no_change": 10,
            }
