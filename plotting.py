"""
Plotting functions — code aligned with psf_demo/psf_tools.py.
Each function can be called independently.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from photutils.profiles import RadialProfile
from photutils.psf import fit_fwhm
from typing import List


def optimal_grid(n_items, max_rows=None):
    if max_rows is None:
        max_rows = n_items
    n = float(n_items)
    best_rows, best_cols = 1, 1
    best_empty = float('inf')
    best_sub = float('inf')
    for rows in range(1, max_rows + 1):
        cols = int(np.ceil(n / rows))
        empty = rows * cols - n_items
        sub = np.abs(cols - rows)
        if sub < best_sub or (sub == best_sub and empty < best_empty):
            best_empty, best_sub = empty, sub
            best_cols, best_rows = cols, rows
    return best_rows, best_cols


def plot_overview(sci_image, valid_coords, valid_ids, outfile,
                  mag_bright_limit=17.0, mag_faint_limit=22.0):
    """Stars marked on full image."""
    sci = fits.getdata(sci_image) if isinstance(sci_image, str) else sci_image
    _, median, std = sigma_clipped_stats(sci, sigma=3.0, maxiters=5)
    norm = simple_norm(sci, 'asinh', vmin=median - 3 * std, vmax=median + 10 * std)
    ny, nx = sci.shape
    fig, ax = plt.subplots(figsize=(10, 10 * ny / nx))
    ax.imshow(sci, norm=norm, origin='lower', cmap='Greys_r')
    x_plot = np.array([c[0] - 1 + 0.5 for c in valid_coords])
    y_plot = np.array([c[1] - 1 + 0.5 for c in valid_coords])
    ax.scatter(x_plot, y_plot, s=50, marker='o', facecolors='none', edgecolors='red',
               label=f'Selected stars ({len(valid_ids)})')
    ax.set_xlabel('X [pixels]', fontsize=12)
    ax.set_ylabel('Y [pixels]', fontsize=12)
    ax.set_title(f'Selected Stars Overview (mag: {mag_bright_limit}-{mag_faint_limit})', fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Overview image saved to {outfile}")


def plot_star_stamps(sci_cutout_list, valid_ids, mask_list, outfile):
    """Grid of individual star cutouts with mask overlay."""
    n = len(sci_cutout_list)
    rows, cols = optimal_grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = np.atleast_1d(axes).flatten()
    for i, ax in enumerate(axes):
        if i < n:
            star = sci_cutout_list[i].copy()
            mask = mask_list[i].copy() if mask_list is not None else None
            norm = simple_norm(star, 'asinh', vmin=np.percentile(star, 5),
                               vmax=np.percentile(star, 98.5))
            ax.imshow(star, origin='lower', cmap='viridis', norm=norm)
            if mask is not None:
                h, w = mask.shape
                mask_overlay = np.zeros((h, w, 4))
                mask_overlay[~mask.astype(bool)] = [0.5, 0.5, 0.5, 0.7]
                ax.imshow(mask_overlay, origin='lower')
            ax.text(0.02, 0.02, f'star{valid_ids[i]}', transform=ax.transAxes,
                    color='white', fontweight='bold', fontsize=6)
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(outfile, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Stars image saved to {outfile}")


def plot_scatter_diagnostics(catalog_file, valid_ids, outfile,
                             elongation_limit=1.2, SNR_limit=100.0,
                             mag_bright_limit=17.0, mag_faint_limit=22.0,
                             pixel_scale=0.262):
    """Scatter plots: SNR vs elongation, mag_auto vs flux_radius*pixel_scale."""
    outtab = ascii.read(catalog_file)
    valid_stars = outtab[np.isin(outtab['label'], valid_ids)]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Left: SNR vs elongation
    ax = axes[0]
    mp = ax.scatter(outtab['segment_flux'] / outtab['segment_fluxerr'], outtab['elongation'],
                    c=outtab['class_star'], cmap='viridis', s=5, alpha=0.5, vmin=0, vmax=1)
    ax.scatter(valid_stars['segment_flux'] / valid_stars['segment_fluxerr'], valid_stars['elongation'],
               s=5, marker='o', facecolors='none', edgecolors='red', alpha=0.7,
               label=f"selected stars ({len(valid_stars)})")
    ax.axvline(x=SNR_limit, ls='--', color='steelblue', lw=1.5, alpha=0.8,
               label=f'SNR = {SNR_limit}')
    ax.axhline(y=elongation_limit, ls='--', color='darkviolet', lw=1.5, alpha=0.8,
               label=f'elong = {elongation_limit}')
    ax.set_xlabel('SNR'); ax.set_ylabel('Elongation')
    snr_vals = outtab['segment_flux'] / outtab['segment_fluxerr']
    ax.set_xscale('log')
    ax.set_xlim(max(snr_vals.min() * 0.5, 0.1), snr_vals.max() * 2.0)
    ax.set_ylim(0.8, 2.8); ax.legend()

    # Right: mag_auto vs flux_radius * pixel_scale
    ax = axes[1]
    ax.scatter(outtab['mag_auto'], outtab['flux_radius'] * pixel_scale,
               c=outtab['class_star'], cmap='viridis', s=5, alpha=0.5, vmin=0, vmax=1)
    ax.scatter(valid_stars['mag_auto'], valid_stars['flux_radius'] * pixel_scale,
               s=5, marker='o', facecolors='none', edgecolors='red', alpha=0.7)
    ax.axvline(x=mag_bright_limit, ls='--', color='crimson', lw=1.5, alpha=0.8,
               label=f'mag = {mag_bright_limit}')
    ax.axvline(x=mag_faint_limit, ls='--', color='dimgray', lw=1.5, alpha=0.8,
               label=f'mag = {mag_faint_limit}')
    ax.set_xlabel('Magnitude'); ax.set_ylabel('flux_radius (arcsec)')
    ax.invert_xaxis(); ax.set_xlim(outtab['mag_auto'].max() + 0.5, outtab['mag_auto'].min() - 0.5); ax.set_yscale('log'); ax.legend()

    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(mp, cax=cbar_ax, label='class_star')
    fig.savefig(outfile, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Stars scatter saved to {outfile}")

def plot_psf_radial_profile(psf, outfile, ax=None, color='red', label=None, pixel_scale=0.262):
    """1D radial profile of PSF (arcsec units)."""
    fwhm_stacked = fit_fwhm(psf)[0]
    label = label + f' (FWHM/2={fwhm_stacked * pixel_scale / 2:.2f} arcsec)' if label else f'PSF Profile (FWHM/2={fwhm_stacked * pixel_scale / 2:.2f} arcsec)'
    max_radius = psf.shape[0] // 2
    psf_radii = np.linspace(0, max_radius, int(max_radius / 1.0) + 1)
    psf_peak_idx = np.argmax(psf)
    psf_peak_center = [psf_peak_idx // psf.shape[1], psf_peak_idx % psf.shape[1]]
    rp_psf = RadialProfile(psf, psf_peak_center, psf_radii)
    psf_profile = rp_psf.profile
    peak_psf = np.nanmax(psf_profile)
    if peak_psf > 0:
        psf_profile = psf_profile / peak_psf

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        own_fig = True
    else:
        own_fig = False
    ax.plot(rp_psf.radius * pixel_scale, psf_profile, color=color, lw=1.2, linestyle='--', label=label)
    ax.axvline(fwhm_stacked * pixel_scale / 2, ls='--', color='black', lw=1.5, alpha=0.7)
    ax.set_xlabel('Radius (arcsec)')
    ax.set_ylabel('Normalized Flux')
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1.2)
    ax.legend()
    ax.set_title('PSF Radial Profile')
    if own_fig:
        fig.savefig(outfile, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"PSF radial profile plot saved to {outfile}")

def plot_radial_profiles(sci_cutout_list, mask_list, psf, outfile, pixel_scale=0.262):
    """1D radial profile: PSF + median star + 16-84 percentile band (arcsec units)."""
    fwhm_stacked = fit_fwhm(psf)[0]
    all_profiles = []
    all_radii = []
    for (star, mask) in zip(sci_cutout_list, mask_list):
        peak_idx = np.argmax(star)
        peak_center = [peak_idx // star.shape[1], peak_idx % star.shape[1]]
        max_radius = star.shape[0] // 2
        star_radii = np.linspace(0, max_radius, int(max_radius / 1.0) + 1)
        rp_star = RadialProfile(star, peak_center, star_radii, mask=~mask.astype(bool))
        star_profile = rp_star.profile
        peak_star = np.nanmax(star_profile)
        if peak_star > 0:
            star_profile = star_profile / peak_star
        all_profiles.append(star_profile)
        all_radii.append(rp_star.radius)
    min_idx = np.argmin([len(p) for p in all_profiles])
    min_len = len(all_profiles[min_idx])
    radii = all_radii[min_idx]
    all_profiles = [p[:min_len] for p in all_profiles]
    all_profiles = np.array(all_profiles)
    median = np.median(all_profiles, axis=0)
    p16 = np.percentile(all_profiles, 16, axis=0)
    p84 = np.percentile(all_profiles, 84, axis=0)

    max_radius = psf.shape[0] // 2
    psf_radii = np.linspace(0, max_radius, int(max_radius / 1.0) + 1)
    psf_peak_idx = np.argmax(psf)
    psf_peak_center = [psf_peak_idx // psf.shape[1], psf_peak_idx % psf.shape[1]]
    rp_psf = RadialProfile(psf, psf_peak_center, psf_radii)
    psf_profile = rp_psf.profile
    peak_psf = np.nanmax(psf_profile)
    if peak_psf > 0:
        psf_profile = psf_profile / peak_psf

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rp_psf.radius * pixel_scale, psf_profile, color='red', lw=2.5, linestyle='--',
            alpha=0.5, label=f'PSF Profile (FWHM/2={fwhm_stacked * pixel_scale / 2:.2f} arcsec)')
    ax.plot(radii * pixel_scale, median, color='black', lw=1.2, label='Median Star Profile')
    ax.fill_between(radii * pixel_scale, p16, p84, color='gray', alpha=0.5, label='16-84th Percentile')
    ax.axvline(fwhm_stacked * pixel_scale / 2, ls='--', color='black', lw=1.5, alpha=0.7)
    ax.set_xlabel('Radius (arcsec)')
    ax.set_ylabel('Normalized Flux')
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1.2)
    ax.legend()
    ax.set_title('Selected Stars Radial Profiles')
    fig.savefig(outfile, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Radial profile plot saved to {outfile}")


def plot_psf_image(psf, outfile=None, ax=None, title=None, cmap='viridis', pixel_scale=0.262):
    """Single PSF image with title showing FWHM.

    If *ax* is given, draw on it and return (does not save).
    Otherwise, create a new figure and save to *output_path*.
    """
    fwhm_stacked = fit_fwhm(psf)[0]
    title = title + f'\nFWHM={fwhm_stacked * pixel_scale:.2f} arcsec' if title else f'Stacked PSF\nFWHM={fwhm_stacked * pixel_scale:.2f} arcsec'

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        own_fig = True
    else:
        own_fig = False

    norm = simple_norm(psf, 'asinh', vmin=np.percentile(psf, 5),
                       vmax=np.percentile(psf, 98.5))
    ax.imshow(psf, origin='lower', cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')

    if own_fig:
        fig.savefig(outfile, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"PSF image saved to {outfile}")


def plot_background(
    sci_orig: np.ndarray,
    bkg_model: np.ndarray,
    sci_sub: np.ndarray,
    source_mask: np.ndarray,
    outfile: str,
):
    """2×2: Original / source mask / background model / subtracted."""
    # Shared stretch for image panels
    vmin = np.percentile(sci_orig, 3)
    vmax = np.percentile(sci_orig, 98)
    norm = simple_norm(sci_orig, 'asinh', vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12),
                             constrained_layout=True)
    fig.get_layout_engine().set(hspace=0.05, wspace=0.02)
    axes = axes.flatten()

    # Panel 1: Original (colorbar left)
    ax = axes[0]
    im = ax.imshow(sci_orig, norm=norm, cmap='Greys', origin='lower')
    ax.set_title(f'Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    fig.colorbar(im, ax=ax, location='left', shrink=0.8)

    # Panel 2: Source mask (no colorbar)
    ax = axes[1]
    ax.imshow(source_mask, cmap='gray', origin='lower', vmin=0, vmax=1)
    n_masked = source_mask.sum()
    ax.set_title(f'Source Mask ({n_masked / source_mask.size * 100:.1f}% masked)',
                 fontsize=12, fontweight='bold')
    ax.axis('off')

    # Panel 3: Background model (colorbar left)
    ax = axes[2]
    im = ax.imshow(bkg_model, norm=norm, cmap='Greys', origin='lower')
    ax.set_title(f'Background Model', fontsize=12, fontweight='bold')
    ax.axis('off')
    fig.colorbar(im, ax=ax, location='left', shrink=0.8)

    # Panel 4: Background-subtracted (colorbar right)
    ax = axes[3]
    im = ax.imshow(sci_sub, norm=norm, cmap='Greys', origin='lower')
    ax.set_title(f'Background-Subtracted', fontsize=12, fontweight='bold')
    ax.axis('off')
    fig.colorbar(im, ax=ax, location='right', shrink=0.8)

    fig.savefig(outfile, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Background comparison saved to {outfile}")


def plot_segment(seg_file, outfile):
    """Quick look at segmentation map."""
    seg = fits.getdata(seg_file)
    fig, ax = plt.subplots(figsize=(8, 8))
    n_labels = len(np.unique(seg)) - 1
    colors = np.random.rand(n_labels + 1, 3)
    colors[0] = [0, 0, 0]  # background black
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    ax.imshow(seg, origin='lower', cmap=cmap, interpolation='nearest')
    ax.set_title('Segmentation Map')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(outfile, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Segment plot saved to {outfile}")


# ── Convenience wrapper ──

def plot_psf_results(
    sci_image,
    catalog_file,
    psf_file,
    valid_ids,
    sci_cutout_list,
    mask_list,
    valid_coords,
    output_dir="./output",
    plot_do_overview=True,
    plot_stars_scatter=True,
    plot_stars_radial_profile=True,
    plot_psf_img=True,
    overview_name="selected_stars_overview.png",
    scatter_name="stars_scatter_diagnostics.png",
    profile_name="stars_radial_profiles.png",
    psfimg_name="stacked_psf.png",
    elongation_limit=1.2,
    mag_bright_limit=17.0,
    mag_faint_limit=22.0,
    SNR_limit=100.0,
    pixel_scale=0.262,
):
    """Convenience wrapper: call all plot functions at once.

    Output filenames can be customised via ``*_name`` parameters.
    """
    psf = fits.getdata(psf_file)
    if plot_do_overview:
        plot_overview(sci_image, valid_coords, valid_ids,
                     os.path.join(output_dir, overview_name),
                     mag_bright_limit, mag_faint_limit)
    if plot_stars_scatter:
        plot_scatter_diagnostics(catalog_file, valid_ids,
                                os.path.join(output_dir, scatter_name),
                                elongation_limit, SNR_limit, mag_bright_limit, mag_faint_limit, pixel_scale)
    if plot_stars_radial_profile:
        plot_radial_profiles(sci_cutout_list, mask_list, psf,
                            os.path.join(output_dir, profile_name), pixel_scale)
    if plot_psf_img:
        plot_psf_image(psf, os.path.join(output_dir, psfimg_name), pixel_scale=pixel_scale)