import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from sklearn.metrics import roc_curve, auc

from base_module import BaseModule
from base_learning_rate import  BaseLR
from gradient_descent import GradientDescent
from learning_rate import FixedLR
from loss_functions import misclassification_error

# from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from modules import L1, L2
from logistic_regression import LogisticRegression
from utils import split_train_test
from cross_validate import cross_validate

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values=[]
    weights=[]

    def callback(val, weight, **kwargs):
        values.append(val)
        weights.append(weight)

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    modules=[L1, L2]
    for module in modules:
        results={}
        min_loss = np.inf
        for eta in etas:
            callback, values, weights= get_gd_state_recorder_callback()

            model= GradientDescent(learning_rate=FixedLR(eta), callback=callback)

            model.fit(module(weights=init), None,None) # ignore X and y
            results[eta]= (values, weights)
            if min(values)<min_loss:
                min_loss=min(values)

            plot_descent_path(module, np.array([init] + weights), title=f"eta={eta}, module={module.__name__}").show()

        fig = go.Figure()
        for eta, (values, weights) in results.items():
            norms = [np.linalg.norm(w) for w in weights]
            fig.add_trace(go.Scatter(x=list(range(len(norms))), y=norms, mode='lines', name=f'eta={eta}'))

        fig.update_layout(
            title=f'Convergence rate for {module.__name__}',
            xaxis_title='GD Iteration',
            yaxis_title='Norm of weights'
        )
        fig.show()

        print(f"The lowest loss achieved when minimizing {module.__name__} is {min_loss}")








def load_data(path: str = "SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    callback, values, weights = get_gd_state_recorder_callback()
    gd= GradientDescent(callback=callback)
    model= LogisticRegression(solver=gd)
    model.fit(X_train.values, y_train.values)
    y_pred= model.predict_proba(X_train.values)


    fpr, tpr, thresholds = roc_curve(y_train, y_pred)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve(area = {roc_auc:.3f})',
                             line=dict(color='darkorange', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random chance',
                             line=dict(color='navy', width=2, dash='dash')))

    fig.update_layout(title='Receiver Operating Characteristic',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      showlegend=True)
    fig.show()

    criterion= tpr-fpr
    optimal_index = np.argmax(criterion)
    model.alpha_ = thresholds[optimal_index]
    print(f"Optimal threshold is {model.alpha_} with a test error of {model.loss(X_test.values, y_test.values)}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas= [0.001,0.002,0.005,0.01,0.02,0.05,0.1]
    best_lambda = None
    best_score = -np.inf

    for lam in lambdas:
        gd= GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000)
        logistic_l1= LogisticRegression(solver=gd, penalty="l1", alpha=0.5, lam=lam)
        train_score, val_score =cross_validate(logistic_l1, X_train.values, y_train.values,
                                               scoring=misclassification_error)

        if val_score > best_score:
            best_score = val_score
            best_lambda = lam

    print(f"Best lambda: {best_lambda}, with score: {best_score}")




if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    #fit_logistic_regression()
