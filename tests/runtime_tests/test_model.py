import numpy as np

def check_model(model):
    assert isinstance(model, dict)
    Ax, Ay, Az = model["Ax"], model["Ay"], model["Az"]
    mins, maxs = model["mins"], model["maxs"]
    assert Ax.shape == Ay.shape == Az.shape == (6,6,6)
    assert mins.shape == maxs.shape == (3,)
    assert np.all(maxs >= mins)
    print("DEBUG: model shapes/keys OK")
