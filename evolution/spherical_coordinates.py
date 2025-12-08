# Code by Alexander Rawlings

import numpy as np
import scipy

def radial_separation(p1, p2=0):
    """
    Determine the radial separation between two particles using scipy.spatial
    function cdist. Note that no unit conversions are performed.

    Parameters
    ----------
    p1 : np.ndarray
        particle 1 position coordinates
    p2 : np.ndarray, float, optional
        particle 2 position coordinates, or a float that specifies the same position along each axis, by default 0 (the origin)

    Returns
    -------
    : np.ndarray
        radial separation (or alternatively, magnitude) between particles
    """
    p1 = np.atleast_2d(p1)
    return scipy.spatial.distance.cdist(p1 - p2, [[0] * p1.shape[-1]]).ravel()


def set_spherical_basis(R):
    """
    Set the spherical coordinate basis.

    Parameters
    ----------
    R : np.ndarray
        array to use to set the spherical coordinate basis, typically will be
        particle position vector

    Returns
    -------
    _r: np.ndarray
        radial component. This will be aligned with the Cartesian direction of
        input R
    _theta: np.ndarray
        angular inclination components orthogonal to _r - Definition as per the
        "physicist's" (ISO) definition
    _phi: np.ndarray
        angular azimuth components orthogonal to _r - Definition as per the
        "physicist's" (ISO) definition
    """
    r = radial_separation(R)  # radial distance
    _r = R / r[:, np.newaxis]  # determine basis vectors
    theta = np.arccos(_r[:, 2])  # arccos(z/r)
    phi = np.arctan2(_r[:, 1], _r[:, 0])  # arctan(y/x)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    _theta = np.stack([cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta], axis=-1)
    _phi = np.stack([-sin_phi, cos_phi, np.zeros_like(phi)], axis=-1)
    return _r, _theta, _phi


def spherical_components(R, v):
    """
    Convert a set of Cartesian values to spherical values.

    Parameters
    ----------
    R : np.ndarray
        Cartesian position coordinates to set spherical basis
    v : np.ndarray
        values to convert to spherical coordinates

    Returns
    -------
    : (n,3) np.ndarray
        spherical components, with columns corresponding to radius, theta, and
        phi
    """
    _r, _theta, _phi = set_spherical_basis(R)
    return np.stack(
        (
            np.sum(_r * v, axis=-1),
            np.sum(_theta * v, axis=-1),
            np.sum(_phi * v, axis=-1),
        ),
        axis=-1,
    )