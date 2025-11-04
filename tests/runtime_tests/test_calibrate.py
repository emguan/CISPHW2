import numpy as np

def check_pred(C_actual, C_pred):

    X = np.asarray(C_actual)
    Y = np.asarray(C_pred)

    avg_err = np.mean(np.abs(X - Y))

    print(f"DEBUG: Average error between C_pred and C_actual: {avg_err:.6f}")