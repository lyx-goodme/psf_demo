"""GALFIT PSF fitting helpers."""

import os, re, subprocess


def write_psf_feedme(
    output_dir,
    input_fits,
    sigma_fits,
    psf_fits,
    mask_fits="none",
    output_fits=None,
    x_init=32.0,
    y_init=32.0,
    mag_init=18.0,
    fit_x=True,
    fit_y=True,
    fit_mag=True,
    zp=22.5,
    plate_scale=0.262,
    cutout_size=63,
    feedme_name="feedme.txt",
):
    """
    Generate a GALFIT feedme file for PSF-only fitting.

    Parameters
    ----------
    output_dir : str
        Directory to save the feedme file (and where galfit output will go).
    input_fits : str
        Path to the star cutout FITS (63x63).
    sigma_fits : str
        Path to the sigma (uncertainty) FITS.
    psf_fits : str
        Path to the PSF FITS.
    mask_fits : str
        Path to mask FITS, or "none".
    x_init, y_init : float
        Initial star position (1-indexed pixel) in the cutout.
    mag_init : float
        Initial integrated magnitude guess.
    fit_x, fit_y, fit_mag : bool
        Whether to fit position and magnitude (1=fit, 0=fixed).
    zp : float
        Magnitude zeropoint.
    plate_scale : float
        Pixel scale in arcsec/pixel.
    cutout_size : int
        Cutout size in pixels (must be odd).
    feedme_name : str
        Output filename.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert to absolute paths — GALFIT may be run from a different cwd
    input_fits  = os.path.abspath(input_fits)
    sigma_fits  = os.path.abspath(sigma_fits) if sigma_fits != "none" else "none"
    psf_fits    = os.path.abspath(psf_fits)
    mask_fits   = os.path.abspath(mask_fits) if mask_fits != "none" else "none"

    if output_fits is None:
        output_fits = os.path.join(output_dir, "output.fits")
    output_fits = os.path.abspath(output_fits)
    feedme_path = os.path.abspath(os.path.join(output_dir, feedme_name))

    x_toggle = 1 if fit_x else 0
    y_toggle = 1 if fit_y else 0
    mag_toggle = 1 if fit_mag else 0

    content = f"""===============================================================================
A) {input_fits}       # Input data image
B) {output_fits}      # Output data image block
C) {sigma_fits}       # Sigma image
D) {psf_fits}         # PSF image
E) 1                  # PSF fine sampling factor
F) {mask_fits}        # Bad pixel mask (1 = bad)
G) none               # Constraints file
H) 1 {cutout_size} 1 {cutout_size}  # Image region to fit
I) {cutout_size} {cutout_size}      # Convolution box
J) {zp}               # Magnitude zeropoint
K) {plate_scale} {plate_scale}      # Plate scale (arcsec/pixel)
O) regular            # Display type
P) 0                  # 0=optimize

 0) psf               # Component type
 1) {x_init:.3f} {y_init:.3f} {x_toggle} {y_toggle}  # Position x, y
 3) {mag_init:.4f} {mag_toggle}          # Integrated magnitude
 Z) 0                 # No more components
==============================================================================="""

    with open(feedme_path, "w") as f:
        f.write(content)

    return feedme_path


def run_galfit(feedme_path, timeout=60, overwrite=True):
    """
    Run GALFIT with the given feedme file.
    Returns the path to the galfit log file.
    """
    fit_dir = os.path.dirname(feedme_path)
    log_path = os.path.join(fit_dir, "galfit.log")

    cmd = ["galfit"]
    if overwrite:
        cmd.append("-o")
    cmd.append(feedme_path)

    with open(log_path, "w") as log:
        subprocess.run(
            cmd,
            cwd=fit_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )

    return log_path


def extract_chi2nu(log_path):
    """
    Extract the final reduced chi-square from a GALFIT log file.
    Returns None if not found.
    """
    params = extract_fitted_params(log_path)
    return params.get("chi2nu")


def extract_fitted_params(log_path):
    """
    Extract the final fitted position and magnitude from a GALFIT log.

    Returns
    -------
    dict with keys: x, y, mag, chi2nu, converged.
    ``converged`` is False when the fit diverged (parameter explosion
    or numerical crash).  Missing / unparseable logs return an empty dict.
    """
    if not os.path.exists(log_path):
        return {}
    with open(log_path) as f:
        text = f.read()

    # Divergence / crash check
    if "Numerical Recipes run-time error" in text:
        return {"converged": False}

    result = {"converged": True}

    # Final chi2nu
    matches = re.findall(r"Chi2nu:\s+([\d\.e\+\-]+)", text)
    if matches:
        chi2nu = float(matches[-1])
        result["chi2nu"] = chi2nu
        if chi2nu > 1e10:
            result["converged"] = False

    # Final fitted parameters (last line with "psf")
    matches = re.findall(
        r"psf\s+:\s+\(\s*([\d\.eE\+\-]+),\s*([\d\.eE\+\-]+)\)\s+([\d\.eE\+\-]+)", text
    )
    if matches:
        x = float(matches[-1][0])
        y = float(matches[-1][1])
        mag = float(matches[-1][2])
        # Check for parameter explosion (position far outside cutout)
        if x < -50 or x > 200 or y < -50 or y > 200:
            result["converged"] = False
        if abs(mag) > 9000:
            result["converged"] = False
        result["x"] = x
        result["y"] = y
        result["mag"] = mag

    return result


def plot_galfit_results(
    data_fits,
    sigma_fits,
    model_fits=None,
    mask_fits=None,
    psf_fits=None,
    outfile=None,
    ax=None,
):
    """Data / model / [PSF] / residual side-by-side.

    If *ax* is given, draw on it and return residual.  Otherwise
    create a new figure and save to *outfile* (if provided).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
    from astropy.visualization import simple_norm
    from matplotlib.colors import TwoSlopeNorm

    if model_fits is None:
        raise ValueError("model_fits or output_dir is required")
    if os.path.isdir(model_fits):
        model_fits = os.path.join(model_fits, "output.fits")

    # --- Pre-flight ---
    if not os.path.exists(model_fits):
        print(f"GALFIT output not found: {model_fits} — skipping plot")
        return None

    try:
        hdul = fits.open(model_fits)
    except Exception as e:
        print(f"GALFIT output corrupt ({e}): {model_fits} — skipping plot")
        return None

    if len(hdul) < 3:
        hdul.close()
        print(f"GALFIT output incomplete ({len(hdul)} HDUs): {model_fits} — skipping plot")
        return None

    galfit_data = hdul[1].data
    model_arr = hdul[2].data
    hdul.close()

    # --- Read data ---
    data = fits.getdata(data_fits)
    residual = galfit_data - model_arr

    # --- Read PSF ---
    psf_img = None
    if psf_fits is not None and os.path.exists(psf_fits):
        psf_img = fits.getdata(psf_fits)

    # --- Read mask ---
    mask = None
    if mask_fits is not None:
        mask = fits.getdata(mask_fits).astype(bool)
        if mask.shape == data.shape:
            mask_overlay = np.zeros((*mask.shape, 4))
            mask_overlay[mask] = [0.5, 0.5, 0.5, 0.5]
        else:
            mask = None

    # --- Setup figure ---
    n_panels = 4 if psf_img is not None else 3
    if ax is not None:
        axes = ax if hasattr(ax, '__iter__') else [ax]
        own_fig = False
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
        own_fig = True

    if psf_img is not None:
        titles = ['Data', 'GALFIT Model', 'PSF Model', 'Residual']
        images = [data, model_arr, psf_img, residual]
        residual_idx = 3
    else:
        titles = ['Data', 'GALFIT Model', 'Residual']
        images = [data, model_arr, residual]
        residual_idx = 2

    # Shared stretch for Data and Model (based on data)
    vmin_data = np.percentile(data, 5)
    vmax_data = np.percentile(data, 98.5)
    data_model_norm = simple_norm(data, 'asinh', vmin=vmin_data, vmax=vmax_data)

    for i, (ax_i, title, img) in enumerate(zip(axes, titles, images)):
        if i == residual_idx:
            vlim = np.nanmax(np.abs(residual)) * 0.8
            vlim = max(vlim, 1e-6)
            norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
            panel_cmap = 'RdBu_r'
        elif i == 0 or i == 1:
            # Data and Model share the same stretch, viridis colormap
            norm = data_model_norm
            panel_cmap = 'viridis'
        else:
            # PSF panel uses its own stretch, viridis
            vmin = np.percentile(img, 5)
            vmax = np.percentile(img, 98.5)
            norm = simple_norm(img, 'asinh', vmin=vmin, vmax=vmax)
            panel_cmap = 'viridis'

        im = ax_i.imshow(img, origin='lower', cmap=panel_cmap, norm=norm)
        ax_i.set_title(title, fontsize=12, fontweight='bold')
        ax_i.axis('off')

        # Mask overlay — only on Data and Model panels
        if i == 0 or i == 1:
            if mask is not None and mask_overlay is not None:
                ax_i.imshow(mask_overlay, origin='lower')

        # Colorbar on residual
        if i == residual_idx:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            cax = inset_axes(ax_i, width='5%', height='100%',
                             loc='lower left',
                             bbox_to_anchor=(1.02, 0., 1, 1),
                             bbox_transform=ax_i.transAxes,
                             borderpad=0)
            cbar = plt.colorbar(im, cax=cax, label='Flux')
            cbar.ax.tick_params(labelsize=8)

        # Stats annotation on residual
        if i == residual_idx:
            res_rms = np.sqrt(np.mean(np.square(residual[np.isfinite(residual)])))
            res_max = np.max(np.abs(residual[np.isfinite(residual)]))
            # Try to read chi2nu from galfit.log
            chi2nu_str = ''
            log_dir = os.path.dirname(model_fits)
            log_path = os.path.join(log_dir, 'galfit.log')
            if os.path.exists(log_path):
                chi2nu = extract_chi2nu(log_path)
                if chi2nu is not None:
                    chi2nu_str = f'\nchi²ν = {chi2nu:.4f}'
            ax_i.text(0.95, 0.05, f'RMS = {res_rms:.4f}{chi2nu_str}\nmax|res| = {res_max:.4f}',
                      transform=ax_i.transAxes, ha='right', va='bottom',
                      color='black', fontsize=9,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    if own_fig:
        fig.subplots_adjust(wspace=0.02, left=0.04, right=0.96)
        if outfile:
            fig.savefig(outfile, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"GALFIT results saved to {outfile}")
        return residual
    else:
        return residual