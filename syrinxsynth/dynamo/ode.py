from numba import jit
import numpy as np

# ode functions:
@jit
def rk4(f, x, h, *args):
    k_1 = f(x, *args) * h
    k_2 = f(x + 0.5 * k_1, *args) * h
    k_3 = f(x + 0.5 * k_2, *args) * h
    k_4 = f(x + k_3, *args) * h
    return x + (k_1 + 2. * (k_2 + k_3) + k_4) / 6.

@jit
def integrate_rk4_fast(vector_field, vector_field_pars: tuple, x_0: np.array, t_0: float, t_f:float, d_t:float, 
steps_per_sample:int=1) -> np.array:
    """
    Integrates a vectorfield using the runge-kutta 4th order method.
    Parameters in the vector field are passed as a tuple of values.
    Arguments:
        vector_field {[type]} -- Function describing the vector field, of the form 
        y = f(x, *args)
        vector_field_pars {tuple} -- parameters of the vector field
        x_0 {np.array} -- initial conditions
        t_0 {float} -- initial time
        t_f {float} -- final time of integration
        d_t {float} -- time step (h in the rk4 function)
        steps_per_sample {int} -- steps to skip when sampling the result (default: 1)

    Returns:
        np.array -- integrated vector field from t_0 to t_f, from initial conditions x_0, 
        sampled at steps_per_sample intervals
    """
    # integrate over a period of time a vector field using rk4
    t = t_0
    x = x_0
    
    n_steps = np.int(np.round((t_f - t) / d_t))
    print(n_steps)
    n_samples = np.int(np.floor(n_steps / steps_per_sample))

    x_out = np.zeros([n_samples, x.shape[0]])
    x_out[:] = np.nan
    
    for i in range(n_steps):
        x = rk4(vector_field, x, d_t, *vector_field_pars)
        t+= d_t
        if not i % steps_per_sample:
            x_out[i // steps_per_sample, :] = x
    return x_out

# vector fields to test:
def harmonic(x, w, w_0):
    return np.array([x[1], - w * w * x[0] / (w_0 * w_0)])


def takens(x, a, b, g):
    gg = g * g
    return np.array([
        x[1],
        a * gg + b * gg * x[0] - gg * x[0] * x[0] * x[0] - g * x[0] * x[0] * x[1] + gg * x[0] * x[0] - g * x[0] * x[1]
    ])


def takens_fast(x, a, b, g):
    gg = g * g
    return np.array([
        x[1],
        gg * (a + x[0] * (b + x[0] * (1. - x[0]))) - g * x[0] * x[1] * (x[0] + 1.)
    ])


# vector fields with parameters in dictionary will be slow (numba don't like)
def takens_dict(x, pars):
    g = pars['gamma']
    gg = g * g

    return np.array([
        x[1],
        gg * (pars['alpha'] + x[0] * (pars['beta'] + x[0] * (1. - x[0]))) - g * x[0] * x[1] * (x[0] + 1.)
    ])


def takens_finch(v, pars):
    g = pars['gamma']
    gg = g * g
    [x, y, i_1, i_2, i_3] = v

    return np.array([
        y,
        pars['alpha_1'] * gg + pars['beta_1'] * gg * x - gg * x * x * x - g * x * x * y + g * x * x - g * x * y,
        i_2,
        -pars['Lg_inv'] * pars['Ch_inv'] * i_1 - pars['Rh'] * (pars['Lb_inv'] + pars['Lg_inv']) * i_2 + (
        pars['Lg_inv'] * pars['Ch_inv'] - pars['Rb'] * pars['Rh'] * pars['Lb_inv'] * pars['Lg_inv']) * i_3
        + pars['Lg_inv'] * pars['dV_ext'] + pars['Rh'] * pars['Lg_inv'] * pars['Lb_inv'] * pars['V_ext'],
        -pars['Lb_inv'] / pars['Lg_inv'] * i_2 - pars['Rb'] * pars['Lb_inv'] * i_3 + pars['Lb_inv'].pars['V_ext']
    ])
