import numpy as np
from scipy.optimize import minimize

def identify_parameters(f, g, h):
    """
    Identifies the parameters a and b for the equation:
    f(t+1)-f(t) = a*(g(t)-f(t)) + b*(h(t)-f(t))
    
    Args:
        f (np.array): Time series f(t)
        g (np.array): Time series g(t)
        h (np.array): Time series h(t)
    
    Returns:
        tuple: (a, b) - the identified parameters
    """
    # Calculate differences
    df = f[1:] - f[:-1]  # f(t+1)-f(t)
    dg = g[:-1] - f[:-1]  # g(t)-f(t)
    dh = h[:-1] - f[:-1]  # h(t)-f(t)
    
    # Define objective function for optimization
    def objective(params):
        a, b = params
        predicted = a * dg + b * dh
        return np.sum((predicted - df) ** 2)  # Sum of squared errors
    
    # Start optimization
    result = minimize(objective, x0=[0.1, 0.1], method='Nelder-Mead')
    
    return result.x[0], result.x[1]

def predict_next_value(f, g, h, a, b):
    """
    Predicts the next value f(t+1) based on the identified parameters.
    
    Args:
        f (np.array): Time series f(t)
        g (np.array): Time series g(t)
        h (np.array): Time series h(t)
        a (float): Identified parameter a
        b (float): Identified parameter b
    
    Returns:
        float: Predicted value f(t+1)
    """
    return f[-1] + a * (g[-1] - f[-1]) + b * (h[-1] - f[-1])

# Example usage:
if __name__ == "__main__":
    # Example data
    t = np.linspace(0, 10, 100)
    f = np.sin(t) + np.random.normal(0, 0.1, 100)
    g = np.cos(t) + np.random.normal(0, 0.1, 100)
    h = np.tan(t) + np.random.normal(0, 0.1, 100)
    
    # Identify parameters
    a, b = identify_parameters(f, g, h)
    print(f"Identified parameters: a = {a:.4f}, b = {b:.4f}")
    
    # Predict next value
    next_value = predict_next_value(f, g, h, a, b)
    print(f"Predicted next value: {next_value:.4f}") 