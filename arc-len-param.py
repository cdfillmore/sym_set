#!/usr/bin/env python3
"""
arc_length_log_spiral.py

Arc-length parameterized logarithmic spiral:
    r(θ) = r0 * exp(b θ)

Arc length formula gives closed-form θ(s), so we can compute
(x(s), y(s)) directly.

Author: ChatGPT
"""
import numpy as np
import matplotlib.pyplot as plt

def theta_from_s(s, r0=1.0, b=0.2):
    """Return θ(s) for arc-length parameter s."""
    return (1.0/b) * np.log(1.0 + (b/(r0*np.sqrt(1+b**2))) * s)

def log_spiral_point(s, r0=1.0, b=0.2):
    """Return (x,y) at arc-length s for the logarithmic spiral."""
    theta = theta_from_s(s, r0, b)
    r = r0 * np.exp(b*theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def log_spiral_points(s_vals, r0=1.0, b=0.2):
    """Vectorized version: arrays of x,y for an array of s values."""
    theta = theta_from_s(s_vals, r0, b)
    r = r0 * np.exp(b*theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

if __name__ == "__main__":
    # Parameters
    plot = False
    r0 = 2.0     # starting radius
    b = 0.3     # growth rate (b>0 means expanding)
    s_max = 120  # arc length to draw
    n_points = 2500

    s_vals = np.linspace(0, s_max, n_points)
    X, Y = log_spiral_points(s_vals, r0=r0, b=b)

    # Verify unit speed: |(dx/ds, dy/ds)| ≈ 1
    dx = np.gradient(X, s_vals)
    dy = np.gradient(Y, s_vals)
    speeds = np.sqrt(dx**2 + dy**2)
    print("speed mean:", np.mean(speeds), "min:", np.min(speeds), "max:", np.max(speeds))

    # Plot
    if plot:
        plt.figure(figsize=(6,6))
        plt.plot(X, Y, linewidth=1.2)
        plt.axis('equal')
        plt.title(f"Arc-length parameterized logarithmic spiral (r0={r0}, b={b})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(alpha=0.3)
        plt.show()
    pts = np.array([X, Y]).T
    simps = np.array([[i,(i+1)%len(pts)] for i in range(len(pts)-1)])
    write_obj("./objs/log_spiral.obj", pts, simps)


############################################################################################
#######     For arc length paramterizing a curve
############################################################################################

#!/usr/bin/env python3
"""
arc_length_param.py

Given a curve as an array of (x,y) points, return an arc-length
parameterized version (uniform sampling in arc length).

Author: ChatGPT
"""
import numpy as np

def arc_length_parameterize(curve, n_points=None, ds=None):
    """
    Reparameterize a polyline curve by arc length.

    Parameters
    ----------
    curve : ndarray of shape (N,2)
        Input curve points (x,y).
    n_points : int, optional
        Number of output points to resample to. If None, must provide ds.
    ds : float, optional
        Step size in arc length. If None, must provide n_points.

    Returns
    -------
    new_curve : ndarray of shape (M,2)
        Resampled curve (arc-length parameterized).
    s_vals : ndarray of shape (M,)
        Arc length coordinates of new points.
    """

    curve = np.asarray(curve)
    if curve.ndim != 2 or curve.shape[1] != 2:
        raise ValueError("curve must be array of shape (N,2)")

    # compute segment lengths and cumulative arc length
    diffs = np.diff(curve, axis=0)
    seg_lens = np.hypot(diffs[:,0], diffs[:,1])
    s = np.concatenate(([0], np.cumsum(seg_lens)))
    total_len = s[-1]

    # choose new sampling
    if n_points is not None:
        s_new = np.linspace(0, total_len, n_points)
    elif ds is not None:
        n_points = int(np.floor(total_len/ds)) + 1
        s_new = np.linspace(0, total_len, n_points)
    else:
        raise ValueError("must provide either n_points or ds")

    # interpolate x(s), y(s)
    x = curve[:,0]
    y = curve[:,1]
    x_new = np.interp(s_new, s, x)
    y_new = np.interp(s_new, s, y)

    new_curve = np.column_stack([x_new, y_new])
    return new_curve, s_new

# -------------------------------
# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example: noisy sine wave curve
    t = np.linspace(0, 4*np.pi, 200)
    x = t
    y = np.sin(t) + 0.1*np.random.randn(len(t))
    curve = np.column_stack([x,y])

    # arc length parameterize to 100 uniform samples
    arc_curve, s_vals = arc_length_parameterize(curve, n_points=300)

    # plot
    plt.figure(figsize=(8,4))
    plt.plot(curve[:,0], curve[:,1], 'k--', alpha=0.5, label="original")
    plt.plot(arc_curve[:,0], arc_curve[:,1], 'r.-', label="arc-length param")
    plt.legend()
    plt.title("Arc-length parameterization of curve")
    plt.show()
