"""
PSF stacking tools for DESI Legacy Survey data.

Pipeline (step by step):
  1. select_valid_stars()        — filter catalog by morphology criteria (no Gaia needed)
  2. subtract_background()       — model and remove sky background
  3. cutout_stars()              — extract stamp cutouts with masks
  4. plot_psf_results()          — visualize overview, stamps, scatter, radial profiles
  5. build_stacked_psf()         — master wrapper calling 1-4 + psfr stacking

Additional helpers:
  gaia_crossmatch()     — crossmatch SEx catalog with Gaia DR3
  apply_gaia_criteria() — add parallax filter on top of morphology criteria
  plot_background()     — show original / background / subtracted side-by-side
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.visualization import simple_norm
from scipy.ndimage import binary_dilation, gaussian_filter
from photutils.background import Background2D, MedianBackground
from photutils.profiles import RadialProfile
from photutils.psf import fit_fwhm
from psfr.psfr import stack_psf
from psfr.util import oversampled2regular
from typing import Optional, Tuple, List


# ═══════════════════════════════════════════════════════════════════
# 1. select_stars — morphology criteria only (no Gaia needed)
# ═══════════════════════════════════════════════════════════════════

def select_valid_stars(
    catalog_file: str,
    seg_file: str,
    elongation_limit: float = 1.2,
    class_star_limit: float = 0.9,
    mag_bright_limit: float = 17.0,
    mag_faint_limit: float = 22.0,
    SNR_limit: float = 100.0,
    cutout_size: int = 63,
    skip_bright_neighbors: bool = True,
    mag_gap_limit: float = 3.0,
    select_ids: List[int] = None,
    exclude_ids: List[int] = None
):
    """
    Filter SExtractor catalog by stellar morphology criteria.
    Returns filtered catalog (astropy Table).
    """
    outtab = ascii.read(catalog_file)
    fltr = (
        (outtab['elongation'] < elongation_limit) &
        (outtab['class_star'] > class_star_limit) &
        (outtab['combined_flags'] == 0) &
        (outtab['mag_auto'] > mag_bright_limit) &
        (outtab['mag_auto'] < mag_faint_limit) &
        (outtab['segment_flux'] / outtab['segment_fluxerr'] > SNR_limit)
    )
    if select_ids is not None:
        fltr &= (np.isin(outtab['label'], select_ids))
        print(f"Input select ids are not None, select valid stars from input ids subsequently.")
    if exclude_ids is not None:
        fltr &= (~np.isin(outtab['label'], exclude_ids))
        print(f"Input exclude ids are not None, exclude the ids for building stacked PSF.")
    outtab1 = outtab[fltr]
    print(f"{len(outtab1)} sources satisfy stellar criteria "
          f"(elong<{elongation_limit}, class_star>{class_star_limit}, "
          f"mag {mag_bright_limit}-{mag_faint_limit}, SNR>{SNR_limit})")

    seg = fits.getdata(seg_file)
    ny, nx = seg.shape
    half = cutout_size // 2

    star_ids = outtab1['label']
    x_image = np.round(outtab1['xcentroid'] + 1, 3)
    y_image = np.round(outtab1['ycentroid'] + 1, 3)
    mag_auto = outtab1['mag_auto']

    valid_ids = []

    for idx, (xc, yc, target_mag) in enumerate(zip(x_image, y_image, mag_auto)):
        target_id = star_ids[idx]
        xc_int = round(xc - 1)
        yc_int = round(yc - 1)
        x_min = xc_int - half
        x_max = xc_int + half + 1
        y_min = yc_int - half
        y_max = yc_int + half + 1

        if x_min >=0 and x_max <= nx and y_min >=0 and y_max <= ny:
            seg_cutout = seg[y_min: y_max, x_min: x_max].copy()
            other_ids = np.unique(seg_cutout[(seg_cutout != 0) & (seg_cutout != target_id)])
            if skip_bright_neighbors:
                skip = False
                for nid in other_ids:
                    other_mag = float(outtab[outtab['label'] == nid]['mag_auto'][0])
                    if other_mag < target_mag + mag_gap_limit:
                        print(f"Source {nid} in the star ID {target_id} cutout is bright or not faint enough "
                              f"(mag={other_mag:.2f}) compared to target star "
                              f"(mag={target_mag:.2f}), skipped.")
                        skip = True
                        break
                if skip:
                    continue
            valid_ids.append(target_id)
        else:
            print(f"Star ID {target_id} at ({xc:.1f}, {yc:.1f}) "
                  f"too close to edge, skipped.")
    if len(valid_ids) == 0:
        raise ValueError(f"No valid stars for stack PSF!")
    result = outtab1[np.isin(outtab1['label'], valid_ids)]
    return result


# ═══════════════════════════════════════════════════════════════════
# 2. subtract_background — model and remove sky background
# ═══════════════════════════════════════════════════════════════════

def subtract_background(
    sci_image: str,
    external_mask: str,
    output_dir: str,
    bkg_box_size: int = 64,
    bkg_filter_size: Tuple[int, int] = (3, 3),
):
    """
    Model and subtract sky background from science image and external mask.
    Returns (sci_sub, bkg_model, bkg_rms, mask).
    """
    sci = fits.getdata(sci_image)
    mask = fits.getdata(external_mask)

    print(f"Modeling and subtracting background "
          f"(box_size={bkg_box_size}, filter_size={bkg_filter_size})...")
    t0 = time.time()

    bkg = Background2D(sci, (bkg_box_size, bkg_box_size),
                       filter_size=bkg_filter_size,
                       mask=mask,
                       bkg_estimator=MedianBackground(),
                       sigma_clip=SigmaClip(sigma=3.0))
    sci_sub = sci - bkg.background

    # Save background model
    hdu_prim = fits.PrimaryHDU()
    hdu_bkg_mask = fits.ImageHDU(mask.astype(np.uint8), name='SOURCE_MASK')
    hdu_bkg = fits.ImageHDU(bkg.background, name='BACKGROUND')
    hdu_rms = fits.ImageHDU(bkg.background_rms, name='BACKGROUND_RMS')
    hdul = fits.HDUList([hdu_prim, hdu_bkg_mask, hdu_bkg, hdu_rms])
    hdul.writeto(os.path.join(output_dir, 'background_model.fits'), overwrite=True)

    print(f"Background subtraction done in {time.time() - t0:.1f}s")
    print(f"Background median: {np.median(bkg.background):.4f}, "
          f"RMS: {np.median(bkg.background_rms):.4f}")

    return sci_sub, bkg.background, bkg.background_rms, mask


# ═══════════════════════════════════════════════════════════════════
# 3. cutout_stars — extract stamps around selected stars
# ═══════════════════════════════════════════════════════════════════

def cutout_stars(
    sci_image,
    seg_file: str,
    catalog_file: str,
    star_ids: List[int],
    output_dir: str = "./stacked_psf_output",
    cutout_size: int = 63,
    save_cutouts: bool = True,
    cutouts_dir: str = "cutouts",
):
    """
    Extract stamp cutouts around selected stars.

    Parameters
    ----------
    sci_image : str or ndarray
        Background-subtracted science image (path or 2D array).
    seg_file : str
        Path to SExtractor segmentation map.
    catalog_file : str
        Path to the FULL SExtractor catalog (used for all star info).
    star_ids : list of int
        Which star IDs (label) to cut out from the catalog.
    ...
    Returns
    -------
    valid_ids, sci_cutout_list, mask_list, valid_coords
    """
    if isinstance(sci_image, str):
        sci = fits.getdata(sci_image)
    else:
        sci = sci_image

    seg = fits.getdata(seg_file)
    half = cutout_size // 2

    # Load full catalog — always use this for all lookups
    full_cat = ascii.read(catalog_file)
    target_ids = set(int(x) for x in star_ids)

    # Filter catalog to target stars only (preserving order from star_ids)
    star_id_order = {int(sid): i for i, sid in enumerate(star_ids)}
    cat_rows = []
    for row in full_cat:
        if int(row['label']) in target_ids:
            cat_rows.append(row)
    # Sort by original star_ids order
    cat_rows.sort(key=lambda r: star_id_order.get(int(r['label']), 999999))

    if len(cat_rows) == 0:
        raise RuntimeError(f"None of the requested star_ids found in catalog.")

    star_labels = [int(r['label']) for r in cat_rows]
    x_image = np.array([r['xcentroid'] + 1 for r in cat_rows])
    y_image = np.array([r['ycentroid'] + 1 for r in cat_rows])

    os.makedirs(output_dir, exist_ok=True)
    cuts_dir = os.path.join(output_dir, cutouts_dir)

    valid_ids = []
    sci_cutout_list = []
    seg_cutout_list = []
    mask_list = []
    valid_coords = []

    for idx, (xc, yc) in enumerate(zip(x_image, y_image)):
        xc_int = round(xc - 1)
        yc_int = round(yc - 1)
        xmin = xc_int - half
        xmax = xc_int + half + 1
        ymin = yc_int - half
        ymax = yc_int + half + 1

        obj_id = star_labels[idx]
        sci_cutout = sci[ymin:ymax, xmin:xmax].copy()
        seg_cutout = seg[ymin:ymax, xmin:xmax].copy()

        nan_mask = np.isnan(sci_cutout)
        sci_cutout[nan_mask] = 0.0

        valid_ids.append(obj_id)
        sci_cutout_list.append(sci_cutout)
        seg_cutout_list.append(seg_cutout)
        valid_coords.append((xc, yc))

        # mask = (seg==0) OR (seg==obj_id) -> True = keep
        # NOTE: psfr uses INVERTED mask logic: 1=keep, 0=reject
        others_mask = (seg_cutout != 0) & (seg_cutout != obj_id)
        others_mask = binary_dilation(others_mask, iterations=2)
        mask = ~others_mask
        mask &= ~nan_mask
        mask_list.append(mask.astype(int))

    n_stars = len(valid_ids)
    print(f"{n_stars} out of {len(x_image)} stars are fully within "
          f"the image and have been cut out.")

    if valid_coords:
        x_valid = np.array([c[0] for c in valid_coords])
        y_valid = np.array([c[1] for c in valid_coords])
        ids = np.array(valid_ids)
        np.savetxt(os.path.join(output_dir, 'star_coordinates.txt'),
                   np.column_stack((x_valid, y_valid, ids)),
                   fmt=['%10.3f', '%10.3f', '%6d'])

    if n_stars == 0:
        raise RuntimeError("No valid stars found within the field of view.")

    # --- Save FITS cutouts (optional) ---
    if save_cutouts:
        os.makedirs(cutouts_dir, exist_ok=True)
        for obj_id, star_data, seg_data, (xc, yc) in zip(
            valid_ids, sci_cutout_list, seg_cutout_list, valid_coords
        ):
            # Science cutout
            hdu = fits.PrimaryHDU(star_data)
            hdu.header['OBJID'] = (obj_id, 'Original ID')
            hdu.header['CUTX'] = (xc, 'Original x centroid (1-indexed)')
            hdu.header['CUTY'] = (yc, 'Original y centroid (1-indexed)')
            hdu.header['SIZE'] = (cutout_size, 'Cutout size in pixels')
            outname = os.path.join(cutouts_dir, f"star{obj_id}.fits")
            hdu.writeto(outname, overwrite=True)
    
            # Segmentation cutout
            hdu_seg = fits.PrimaryHDU(seg_data.astype(np.int32))
            hdu_seg.header['OBJID'] = (obj_id, 'Original SExtractor ID')
            hdu_seg.header['CUTX'] = (xc, 'Original x centroid (1-indexed)')
            hdu_seg.header['CUTY'] = (yc, 'Original y centroid (1-indexed)')
            hdu_seg.header['SIZE'] = (cutout_size, 'Cutout size in pixels')
            outname_seg = os.path.join(cutouts_dir, f"star{obj_id}_seg.fits")
            hdu_seg.writeto(outname_seg, overwrite=True)
        print(f"Cropped star images saved to {cutouts_dir}/")

    return valid_ids, sci_cutout_list, mask_list, valid_coords


# ═══════════════════════════════════════════════════════════════════
# 4. plot_psf_results — visualize everything
# ═══════════════════════════════════════════════════════════════════

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


def plot_psf_results(
    catalog_file: str,
    psf_file: str,
    valid_ids: List,
    sci_cutout_list: List,
    mask_list: List,
    output_dir: str = "./stacked_psf_output",
    plot_stars_image: bool = True,
    plot_stars_scatter: bool = True,
    plot_stars_radial_profile: bool = True,
    plot_psf_image: bool = True,
    elongation_limit: float = 1.2,
    mag_bright_limit: float = 17.0,
    mag_faint_limit: float = 22.0,
    SNR_limit: float = 100.0,
    pixel_scale: float = 0.262,
):
    outtab = ascii.read(catalog_file)
    psf = fits.getdata(psf_file)
    fwhm_stacked = fit_fwhm(psf)[0]

    # --- Star stamps ---
    if plot_stars_image:
        n = len(sci_cutout_list)
        rows, cols = optimal_grid(n)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten()
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
                        color='white', fontweight='bold')
            ax.axis('off')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'selected_stars_image.png'),
                    dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f"Stars image saved to {output_dir}/selected_stars_image.png")

    # --- Scatter ---
    if plot_stars_scatter:
        valid_stars = outtab[np.isin(outtab['label'], valid_ids)]
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        ax = axes[0]
        mp = ax.scatter(outtab['segment_flux'] / outtab['segment_fluxerr'], outtab['elongation'],
                        c=outtab['class_star'], cmap='viridis', s=5, alpha=0.5, vmin=0, vmax=1)
        ax.scatter(valid_stars['segment_flux'] / valid_stars['segment_fluxerr'], valid_stars['elongation'],
                   s=5, marker='o', facecolors='none', edgecolors='red', alpha=0.7,
                   label=f"selected stars ({len(valid_stars)})")
        ax.axvline(x=SNR_limit, ls='--', color='red', lw=1.5, alpha=0.8)
        ax.axhline(y=elongation_limit, ls='--', color='red', lw=1.5, alpha=0.8)
        ax.set_xlabel('SNR'); ax.set_ylabel('Elongation')
        ax.set_xscale('log'); ax.set_ylim(0.8, 2.8); ax.legend()
        ax = axes[1]
        ax.scatter(outtab['mag_auto'], outtab['kron_radius'] * pixel_scale,
                   c=outtab['class_star'], cmap='viridis', s=5, alpha=0.5, vmin=0, vmax=1)
        ax.scatter(valid_stars['mag_auto'], valid_stars['kron_radius'] * pixel_scale,
                   s=5, marker='o', facecolors='none', edgecolors='red', alpha=0.7)
        ax.axvline(x=mag_bright_limit, ls='--', color='red', lw=1.5, alpha=0.8)
        ax.axvline(x=mag_faint_limit, ls='--', color='red', lw=1.5, alpha=0.8)
        ax.set_xlabel('Magnitude'); ax.set_ylabel('Kron Radius (arcsec)')
        ax.invert_xaxis(); ax.set_ylim(0.8, 1.8)
        plt.tight_layout()
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        fig.colorbar(mp, cax=cbar_ax, label='class_star')
        fig.savefig(os.path.join(output_dir, 'stars_scatter.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Stars scatter saved to {output_dir}/stars_scatter.png")

    # --- Radial profiles ---
    if plot_stars_radial_profile:
        all_profiles = []
        all_radii = []
        for star, mask in zip(sci_cutout_list, mask_list):
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
                alpha=0.5, label=f'PSF Profile (FWHM/2={fwhm_stacked / 2:.2f} pix)')
        ax.plot(radii * pixel_scale, median, color='black', lw=1.2, label='Median Star Profile')
        ax.fill_between(radii * pixel_scale, p16, p84, color='gray', alpha=0.5, label='16-84th Percentile')
        ax.axvline(fwhm_stacked * pixel_scale / 2, ls='--', color='black', lw=1.5, alpha=0.7)
        ax.set_xlabel('Radius (arcsec)')
        ax.set_ylabel('Normalized Flux')
        ax.set_yscale('log')
        ax.legend()
        ax.set_title('Selected Stars Radial Profiles')
        fig.savefig(os.path.join(output_dir, 'stars_radial_profiles.png'),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Radial profile plot saved to {output_dir}/stars_radial_profiles.png")

    # --- PSF image ---
    if plot_psf_image:
        fig, ax = plt.subplots(figsize=(6, 6))
        norm = simple_norm(psf, 'asinh', vmin=np.percentile(psf, 5), vmax=np.percentile(psf, 99.5))
        ax.imshow(psf, origin='lower', cmap='Greys_r', norm=norm)
        ax.set_title(f'Stacked PSF\\nFWHM={fwhm_stacked * pixel_scale:.2f} arcsec',
                     fontsize=12, fontweight='bold')
        ax.axis('off')
        fig.savefig(os.path.join(output_dir, 'stacked_psf.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"PSF image saved to {output_dir}/stacked_psf.png")


def plots_to_pdf(output_dir: str, pdf_name: str = "results.pdf"):
    """Combine all PNG files in output_dir into a single PDF."""
    from matplotlib.backends.backend_pdf import PdfPages
    import glob

    png_files = sorted(glob.glob(os.path.join(output_dir, '*.png')))
    if not png_files:
        print("No PNG files found to combine into PDF.")
        return

    pdf_path = os.path.join(output_dir, pdf_name)
    with PdfPages(pdf_path) as pdf:
        for png_file in png_files:
            img = plt.imread(png_file)
            fig, ax = plt.subplots(figsize=(img.shape[1]/100, img.shape[0]/100))
            ax.imshow(img)
            ax.axis('off')
            fig.tight_layout(pad=0)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    print(f"Combined {len(png_files)} plots into {pdf_path}")


# ═══════════════════════════════════════════════════════════════════
# 5. build_stacked_psf — master pipeline
# ═══════════════════════════════════════════════════════════════════

def build_stacked_psf(
    sci_image: str,
    catalog_file: str,
    seg_file: str,
    output_dir: str = "./stacked_psf_output",
    # Selection criteria
    elongation_limit: float = 1.5,
    class_star_limit: float = 0.9,
    mag_bright_limit: float = 19.0,
    mag_faint_limit: float = 21.0,
    SNR_limit: float = 100.0,
    select_ids=None,         # pre-selected IDs (skip select_stars if provided)
    exclude_ids=None,       # IDs to exclude from selection
    skip_bright_neighbors: bool = True,
    mag_gap_limit: float = 3.0,
    # Cutout
    cutout_size: int = 71,
    save_cutouts: bool = True,
    cutouts_dir: str = "cutouts",
    # Background
    subtract_bkg: bool = False,
    external_mask: Optional[str] = None,  # required if subtract_bkg=True
    bkg_box_size: int = 64,
    bkg_filter_size: Tuple[int, int] = (3, 3),
    # Stacking
    oversampling: int = 3,
    n_recenter: int = 10,
    num_iteration: int = 20,
    # Plotting
    pixel_scale: float = 0.262,
    plot_stars_image: bool = False,
    plot_stars_scatter: bool = False,
    plot_stars_radial_profile: bool = False,
    plot_psf_image: bool = False,
    # Output
    psf_output_name: str = "psf.fits",
):
    """
    Full PSF stacking pipeline.

    Steps:
      1. select_valid_stars()       — filter catalog (skipped if star_ids provided)
      2. subtract_background()      — remove sky background (if subtract_bkg=True)
      3. cutout_stars()             — extract stamp cutouts
      4. stack_psf (psfr)           — build oversampled PSF
      5. plot_psf_results()         — visualize
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Select stars (or use provided IDs)
    selected = select_valid_stars(
            catalog_file=catalog_file,
            seg_file=seg_file,
            elongation_limit=elongation_limit,
            class_star_limit=class_star_limit,
            mag_bright_limit=mag_bright_limit,
            mag_faint_limit=mag_faint_limit,
            SNR_limit=SNR_limit,
            cutout_size=cutout_size,
            skip_bright_neighbors=skip_bright_neighbors,
            mag_gap_limit=mag_gap_limit,
            select_ids=select_ids,
            exclude_ids=exclude_ids,
        )
    target_ids = [int(x) for x in selected['label']]

    # Step 2: Background subtraction (optional, requires external_mask)
    if subtract_bkg:
        if external_mask is None:
            raise ValueError("external_mask is required when subtract_bkg=True")
        sci_sub, bkg_model, bkg_rms, source_mask = subtract_background(
            sci_image, external_mask, output_dir,
            bkg_box_size=bkg_box_size, bkg_filter_size=bkg_filter_size,
        )
    else:
        sci_sub = fits.getdata(sci_image) if isinstance(sci_image, str) else sci_image  # use original image as-is

    # Step 3: Cutout stars
    valid_ids, sci_cutout_list, mask_list, valid_coords = cutout_stars(
        sci_image=sci_sub,
        seg_file=seg_file,
        catalog_file=catalog_file,
        star_ids=target_ids,
        output_dir=output_dir,
        cutout_size=cutout_size,
        save_cutouts=save_cutouts,
        cutouts_dir=cutouts_dir,
    )

    # Step 4: Stack PSF
    result = stack_psf(
        sci_cutout_list,
        oversampling=oversampling,
        mask_list=mask_list,
        error_map_list=None,
        saturation_limit=None,
        num_iteration=num_iteration,
        n_recenter=n_recenter,
    )
    psf_oversampled = result[0]
    psf = oversampled2regular(psf_oversampled, oversampling)
    psf_out = os.path.join(output_dir, psf_output_name)
    fits.writeto(psf_out, psf, overwrite=True)

    # Step 5: Plot results
    plot_psf_results(
        catalog_file=catalog_file,
        psf_file=psf_out,
        valid_ids=valid_ids,
        sci_cutout_list=sci_cutout_list,
        mask_list=mask_list,
        output_dir=output_dir,
        plot_stars_image=plot_stars_image,
        plot_stars_scatter=plot_stars_scatter,
        plot_stars_radial_profile=plot_stars_radial_profile,
        plot_psf_image=plot_psf_image,
        elongation_limit=elongation_limit,
        mag_bright_limit=mag_bright_limit,
        mag_faint_limit=mag_faint_limit,
        SNR_limit=SNR_limit,
        pixel_scale=pixel_scale,
    )

    # Step 6: Combine plots into PDF
    plots_to_pdf(output_dir)

    return psf_out


# ═══════════════════════════════════════════════════════════════════
# Additional helpers for Gaia crossmatch
# ═══════════════════════════════════════════════════════════════════

def gaia_crossmatch(catalog_file: str, max_distance: float = 0.5):
    """
    Crossmatch SEx catalog with Gaia DR3 via CDS XMatch.
    Returns pandas DataFrame with merged columns.
    """
    from astroquery.xmatch import XMatch
    from astropy import units as u

    cat = ascii.read(catalog_file)
    gaia = XMatch.query(
        cat1=cat,
        cat2='vizier:I/355/gaiadr3',
        max_distance=max_distance * u.arcsec,
        colRA1='ra',
        colDec1='dec'
    )
    df = gaia.to_pandas()
    print(f"Gaia crossmatch: {len(df)}/{len(cat)} sources matched")
    return df


def apply_gaia_criteria(
    gaia_df: pd.DataFrame,
    elongation_limit: float = 1.2,
    class_star_limit: float = 0.9,
    mag_bright_limit: float = 17.0,
    mag_faint_limit: float = 22.0,
    SNR_limit: float = 100.0,
    plx_snr_limit: float = 3.0,
):
    """
    Filter Gaia-crossmatched catalog by morphology + parallax criteria.
    Returns filtered DataFrame.
    """
    mask = (
        (gaia_df['elongation'] < elongation_limit) &
        (gaia_df['class_star'] > class_star_limit) &
        (gaia_df['combined_flags'] == 0) &
        (gaia_df['mag_auto'] > mag_bright_limit) &
        (gaia_df['mag_auto'] < mag_faint_limit) &
        (gaia_df['segment_flux'] / gaia_df['segment_fluxerr'] > SNR_limit) &
        (gaia_df['Plx'] / gaia_df['e_Plx'] > plx_snr_limit)
    )
    result = gaia_df[mask].reset_index(drop=True)
    print(f"{len(result)} stars after all criteria "
          f"(incl. Plx/e_Plx > {plx_snr_limit})")
    return result


def plot_background(
    sci_orig: np.ndarray,
    bkg_model: np.ndarray,
    sci_sub: np.ndarray,
    source_mask: np.ndarray,
    output_dir: str,
    band: str = "r",
):
    """Plot original, background model, and subtracted images side-by-side.
    All three panels share the same asinh stretch (from original image)
    with individual colorbars, matching psf_test.ipynb style."""
    vmin = np.percentile(sci_orig, 0.03)
    vmax = np.percentile(sci_orig, 99)
    norm = simple_norm(sci_orig, 'asinh', vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # Original
    ax = axes[0]
    im0 = ax.imshow(sci_orig, norm=norm, cmap='viridis', origin='lower')
    ax.imshow(source_mask, cmap='gray', alpha=0.3, origin='lower', vmin=0, vmax=1)
    ax.axis('off')
    ax.set_title(f'Original Image ({band} band)', fontsize=12, fontweight='bold')
    fig.colorbar(im0, ax=ax, orientation='horizontal', pad=0.01, shrink=0.8)

    # Background model (same stretch as original for comparison)
    ax = axes[1]
    im1 = ax.imshow(bkg_model, norm=norm, cmap='viridis', origin='lower')
    ax.axis('off')
    ax.set_title(f'Background Model ({band} band)', fontsize=12, fontweight='bold')
    fig.colorbar(im1, ax=ax, orientation='horizontal', pad=0.01, shrink=0.8)

    # Subtracted
    ax = axes[2]
    im2 = ax.imshow(sci_sub, norm=norm, cmap='viridis', origin='lower')
    ax.imshow(source_mask, cmap='gray', alpha=0.3, origin='lower', vmin=0, vmax=1)
    ax.axis('off')
    ax.set_title(f'Background-Subtracted ({band} band)', fontsize=12, fontweight='bold')
    fig.colorbar(im2, ax=ax, orientation='horizontal', pad=0.01, shrink=0.8)

    out = os.path.join(output_dir, 'background_subtraction.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Background comparison saved to {out}")