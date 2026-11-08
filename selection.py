"""
PSF stacking tools for DESI Legacy Survey data.

Core
----
select_valid_stars()       — morphology + boundary + neighbor filtering
cutout_stars()             — extract stamp cutouts with masks
build_psf()                — stack PSF via psfr
stacked_psf_pipeline()     — select → subtract → cutout → build PSF → plot

Background
----------
subtract_background()      — sky subtraction (uses external mask HDU[1])
generate_background_mask() — create source mask via SExtractor seg + dilation

Gaia
----
gaia_crossmatch()          — CDS XMatch or ESA ADQL + local k-d tree
"""

import os, time
import numpy as np
import pandas as pd
from astropy.io import fits, ascii
from astropy.stats import SigmaClip, sigma_clipped_stats
from scipy.ndimage import binary_dilation, gaussian_filter
from photutils.background import Background2D, MedianBackground
from psfr.psfr import stack_psf
from psfr.util import oversampled2regular
from typing import Optional, Tuple, List

from .plotting import plot_psf_results


# ═══════════════════════════════════════════════════════════════════
# 1. select_valid_stars
# ═══════════════════════════════════════════════════════════════════

def select_valid_stars(
    catalog_file: str,
    seg_file: str,
    elongation_limit: float = 1.2,
    class_star_limit: float = 0.9,
    combined_flags_limit: int = 1,
    mag_bright_limit: float = 17.0,
    mag_faint_limit: float = 22.0,
    SNR_limit: float = 0.0,
    cutout_size: int = 63,
    skip_bright_neighbors: bool = True,
    mag_gap_limit: float = 3.0,
    exclude_ids: Optional[List[int]] = None,
    select_ids: Optional[List[int]] = None,
    verbose: bool = False,
):
    """
    Filter catalog by morphology, boundary, and neighbor checks.

    Returns filtered astropy Table.
    """
    outtab = ascii.read(catalog_file)
    fltr = (
        (outtab['elongation'] < elongation_limit) &
        (outtab['class_star'] > class_star_limit) &
        (outtab['combined_flags'] < combined_flags_limit) &
        (outtab['mag_auto'] > mag_bright_limit) &
        (outtab['mag_auto'] < mag_faint_limit) &
        (outtab['segment_flux'] / outtab['segment_fluxerr'] > SNR_limit)
    )
    if select_ids is not None:
        fltr &= np.isin(outtab['label'], select_ids)
    if exclude_ids is not None:
        fltr &= ~np.isin(outtab['label'], exclude_ids)
    outtab1 = outtab[fltr]

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
        xc_int, yc_int = round(xc - 1), round(yc - 1)
        x_min, x_max = xc_int - half, xc_int + half + 1
        y_min, y_max = yc_int - half, yc_int + half + 1

        if not (x_min >= 0 and x_max <= nx and y_min >= 0 and y_max <= ny):
            if verbose:
                print(f"Star ID {target_id} at ({xc:.1f}, {yc:.1f}) "
                      f"too close to edge, skipped.")
            continue

        if skip_bright_neighbors:
            seg_cut = seg[y_min:y_max, x_min:x_max]
            other_ids = np.unique(seg_cut[(seg_cut != 0) & (seg_cut != target_id)])
            skip = False
            for nid in other_ids:
                rows = outtab[outtab['label'] == nid]
                if len(rows) > 0:
                    if float(rows['mag_auto'][0]) < target_mag + mag_gap_limit:
                        if verbose:
                            print(f"Source {nid} in the star ID {target_id} cutout is "
                                  f"bright (mag={float(rows['mag_auto'][0]):.2f}) vs "
                                  f"target (mag={target_mag:.2f}), skipped.")
                        skip = True
                        break
            if skip:
                continue
        valid_ids.append(target_id)

    if len(valid_ids) == 0:
        raise ValueError("No valid stars found.")
    result = outtab1[np.isin(outtab1['label'], valid_ids)]
    print(f"{len(result)} sources satisfy stellar criteria "
          f"(elong<{elongation_limit}, class_star>{class_star_limit}, "
          f"combined_flags<{combined_flags_limit}, "
          f"mag {mag_bright_limit}-{mag_faint_limit}, SNR>{SNR_limit})")
    return result


# ═══════════════════════════════════════════════════════════════════
# 2. subtract_background
# ═══════════════════════════════════════════════════════════════════

def subtract_background(
    sci_image,
    external_mask,
    output_dir: str = None,
    bkg_box_size: int = 64,
    bkg_filter_size: Tuple[int, int] = (3, 3),
):
    """
    Model and subtract sky background.  Returns (sci_sub, bkg_model, bkg_rms, mask).

    Parameters
    ----------
    sci_image : str or ndarray
        Science image (path or 2D array).
    external_mask : str or ndarray
        Source mask (path, or 2D bool/uint8 array).  If a file path, reads HDU[0].
    output_dir : str, optional
        If given, save ``background_model.fits`` (HDUList with mask, bkg, rms).
    """
    sci = fits.getdata(sci_image) if isinstance(sci_image, str) else sci_image
    if isinstance(external_mask, str):
        mask = fits.getdata(external_mask)
    else:
        mask = np.asarray(external_mask, dtype=bool)

    print(f"Modeling background (box={bkg_box_size}, filter={bkg_filter_size})...")
    t0 = time.time()

    bkg = Background2D(sci, (bkg_box_size, bkg_box_size),
                       filter_size=bkg_filter_size,
                       mask=mask,
                       bkg_estimator=MedianBackground(),
                       sigma_clip=SigmaClip(sigma=3.0))
    sci_sub = sci - bkg.background

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        hdu_prim = fits.PrimaryHDU(data=np.zeros((1,), dtype=np.uint8))
        hdu_bkg_mask = fits.ImageHDU(mask.astype(np.uint8), name='SOURCE_MASK')
        hdu_bkg = fits.ImageHDU(bkg.background, name='BACKGROUND')
        hdu_rms = fits.ImageHDU(bkg.background_rms, name='BACKGROUND_RMS')
        hdul = fits.HDUList([hdu_prim, hdu_bkg_mask, hdu_bkg, hdu_rms])
        hdul.writeto(os.path.join(output_dir, 'background_model.fits'), overwrite=True)

    print(f"  Done in {time.time() - t0:.1f}s, bkg median={np.median(bkg.background):.4f}, "
          f"RMS={np.median(bkg.background_rms):.4f}")
    return sci_sub, bkg.background, bkg.background_rms, mask


# ═══════════════════════════════════════════════════════════════════
# 3. cutout_stars
# ═══════════════════════════════════════════════════════════════════

def cutout_stars(
    sci_image,
    seg_file: str,
    catalog_file: str,
    star_ids: List[int],
    output_dir: str = "./output",
    cutout_size: int = 63,
    save_cutouts: bool = True,
    cutouts_dir: str = "cutouts",
    sigma_image = None,
    sigma_bad_mask = None,
):
    """
    Extract stamp cutouts for the given star IDs.

    Parameters
    ----------
    sci_image : str or ndarray
        Background-subtracted image.
    sigma_image : str or ndarray, optional
        Sigma (uncertainty) map. If given, sigma stamps are cut and saved
        alongside science stamps as ``star{id}_sigma.fits``.
    catalog_file : str
        Full SEx catalog (for coordinates).
    star_ids : list of int
        Which stars to cut out.
    """
    if isinstance(sci_image, str):
        sci = fits.getdata(sci_image)
    else:
        sci = sci_image

    seg = fits.getdata(seg_file)

    # Load sigma if provided
    if sigma_image is not None:
        if isinstance(sigma_image, str):
            sig_full = fits.getdata(sigma_image)
        else:
            sig_full = sigma_image
    else:
        sig_full = None

    half = cutout_size // 2

    full_cat = ascii.read(catalog_file)
    target_ids = set(int(x) for x in star_ids)

    # Order by star_ids input
    cat_rows = []
    for row in full_cat:
        if int(row['label']) in target_ids:
            cat_rows.append(row)
    id_order = {int(s): i for i, s in enumerate(star_ids)}
    cat_rows.sort(key=lambda r: id_order.get(int(r['label']), 999999))

    if len(cat_rows) == 0:
        raise RuntimeError("None of the requested star IDs found in catalog.")

    os.makedirs(output_dir, exist_ok=True)
    cuts_dir = os.path.join(output_dir, cutouts_dir)

    valid_ids, sci_cuts, seg_cuts, sig_cuts, mask_list, valid_coords = [], [], [], [], [], []
    for row in cat_rows:
        obj_id = int(row['label'])
        xc = float(row['xcentroid']) + 1
        yc = float(row['ycentroid']) + 1
        xci, yci = round(xc - 1), round(yc - 1)
        xmin, xmax = xci - half, xci + half + 1
        ymin, ymax = yci - half, yci + half + 1

        sci_cut = sci[ymin:ymax, xmin:xmax].copy()
        seg_cut = seg[ymin:ymax, xmin:xmax].copy()
        sig_cut = sig_full[ymin:ymax, xmin:xmax].copy() if sig_full is not None else None

        nan_mask = np.isnan(sci_cut)
        sci_cut[nan_mask] = 0.0

        others_mask = (seg_cut != 0) & (seg_cut != obj_id)
        others_mask = binary_dilation(others_mask, iterations=2)
        mask = (~others_mask) & (~nan_mask)

        # Extend mask to exclude sigma<=0 pixels (no valid data)
        # Use original sigma_bad_mask when available (before bkg_rms propagation)
        # to catch pixels where sigma_orig=0 even if sigma_eff>0 from bkg_rms
        if sigma_bad_mask is not None:
            bad_cut = sigma_bad_mask[ymin:ymax, xmin:xmax]
            n_bad = bad_cut.sum()
            if n_bad > 0:
                if n_bad > 50:
                    print(f"    star{obj_id}: {n_bad} zero-weight pixels added to mask "
                          f"({100*n_bad/bad_cut.size:.1f}% of cutout)")
                mask = mask & (~bad_cut)

        valid_ids.append(obj_id)
        sci_cuts.append(sci_cut)
        seg_cuts.append(seg_cut)
        mask_list.append(mask.astype(int))
        valid_coords.append((xc, yc))
        if sig_cut is not None:
            sig_cuts.append(sig_cut)

    print(f"  Cutout: {len(valid_ids)} stamps")

    if save_cutouts:
        os.makedirs(cuts_dir, exist_ok=True)
        for i_cut, (obj_id, sc, sg, mk) in enumerate(
                zip(valid_ids, sci_cuts, seg_cuts, mask_list)):
            xc, yc = valid_coords[i_cut]
            # Science cutout
            hdu = fits.PrimaryHDU(sc)
            hdu.header['OBJID'] = obj_id
            hdu.header['CUTX'] = (xc, 'Original x centroid (1-indexed)')
            hdu.header['CUTY'] = (yc, 'Original y centroid (1-indexed)')
            hdu.writeto(os.path.join(cuts_dir, f"star{obj_id}.fits"), overwrite=True)
            # Segmentation cutout
            hdu_seg = fits.PrimaryHDU(sg.astype(np.int32))
            hdu_seg.header['OBJID'] = obj_id
            hdu_seg.writeto(os.path.join(cuts_dir, f"star{obj_id}_seg.fits"), overwrite=True)
            # Mask (True = keep, False = reject)
            hdu_msk = fits.PrimaryHDU(mk.astype(np.uint8))
            hdu_msk.header['OBJID'] = obj_id
            hdu_msk.header['NOTE'] = ('1=keep, 0=reject', 'psfr convention')
            hdu_msk.writeto(os.path.join(cuts_dir, f"star{obj_id}_mask.fits"), overwrite=True)
            # Sigma stamp
            if sig_cuts:
                hdu_sig = fits.PrimaryHDU(sig_cuts[i_cut].astype(np.float32))
                hdu_sig.header['OBJID'] = obj_id
                hdu_sig.writeto(os.path.join(cuts_dir, f"star{obj_id}_sigma.fits"), overwrite=True)
        print(f"    Saved to {cuts_dir}/")

    return valid_ids, sci_cuts, mask_list, valid_coords, sig_cuts


# ═══════════════════════════════════════════════════════════════════
# 4. build_psf — stack PSF from cutouts
# ═══════════════════════════════════════════════════════════════════

def build_psf(
    sci_cuts: list,
    mask_list: Optional[list] = None,
    error_list: Optional[list] = None,
    oversampling: int = 3,
    num_iteration: int = 20,
    n_recenter: int = 10,
):
    """
    Build a stacked PSF from cutouts using psfr.

    Returns (psf_regular, psf_oversampled).
    """
    result = stack_psf(
        sci_cuts,
        oversampling=oversampling,
        mask_list=mask_list,
        error_map_list=error_list,
        saturation_limit=None,
        num_iteration=num_iteration,
        n_recenter=n_recenter,
    )
    psf_oversampled = result[0]
    psf = oversampled2regular(psf_oversampled, oversampling)
    return psf, psf_oversampled


# ═══════════════════════════════════════════════════════════════════
# 5. stacked_psf_pipeline — convenience wrapper
# ═══════════════════════════════════════════════════════════════════

def stacked_psf_pipeline(
    sci_image: str,
    catalog_file: str,
    seg_file: str,
    output_dir: str = "./output",
    elongation_limit: float = 1.2,
    class_star_limit: float = 0.9,
    mag_bright_limit: float = 17.0,
    mag_faint_limit: float = 22.0,
    SNR_limit: float = 0.0,
    cutout_size: int = 63,
    oversampling: int = 3,
    subtract_bkg: bool = False,
    external_mask: Optional[str] = None,
    sigma_image: Optional[str] = None,
    star_ids: Optional[List[int]] = None,
    psf_output_name: str = "psf.fits",
    # Plotting flags
    plot_stamps: bool = True,
    plot_radial: bool = True,
    plot_psf_img: bool = True,
    plot_scatter: bool = True,
    plot_over: bool = True,
    pixel_scale: float = 0.262,
    overview_name: str = "selected_stars_overview.png",
    stamps_name: str = "selected_stars_image.png",
    scatter_name: str = "stars_scatter_diagnostics.png",
    profile_name: str = "stars_radial_profiles.png",
    psfimg_name: str = "stacked_psf.png",
):
    """
    Full pipeline: select → subtract → cutout → build PSF → plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Select
    if star_ids is not None:
        target_ids = [int(x) for x in star_ids]
        selected = None
    else:
        selected = select_valid_stars(
            catalog_file, seg_file=seg_file,
            elongation_limit=elongation_limit, class_star_limit=class_star_limit,
            mag_bright_limit=mag_bright_limit, mag_faint_limit=mag_faint_limit,
            SNR_limit=SNR_limit, cutout_size=cutout_size)
        target_ids = [int(x) for x in selected['label']]

    # 2. Background
    if subtract_bkg:
        if external_mask is None:
            raise ValueError("external_mask required for background subtraction")
        sci_sub, bkg_model, bkg_rms, _ = subtract_background(
            sci_image, external_mask, output_dir)
    else:
        sci_sub = sci_image
        bkg_rms = None

    # Propagate background-subtraction uncertainty into sigma
    sigma_bad = None
    if sigma_image is not None and bkg_rms is not None:
        sigma_orig = fits.getdata(sigma_image)
        sigma_bad = (sigma_orig <= 0)  # original zero-weight pixels
        sigma_eff = np.sqrt(sigma_orig.astype(float)**2 + bkg_rms.astype(float)**2)
        print(f"Sigma propagated: sqrt(sigma² + bkg_rms²); "
              f"sigma median={np.median(sigma_orig):.4f}, "
              f"bkg_rms median={np.median(bkg_rms):.4f}, "
              f"sigma_eff median={np.median(sigma_eff):.4f}")
    elif sigma_image is not None:
        sigma_eff = sigma_image
        sigma_orig = fits.getdata(sigma_image)
        sigma_bad = (sigma_orig <= 0)
    else:
        sigma_eff = None

    # 3. Cutout
    valid_ids, sci_cuts, mask_list, valid_coords, sig_cuts = cutout_stars(
        sci_sub, seg_file, catalog_file, target_ids,
        output_dir=output_dir, cutout_size=cutout_size,
        sigma_image=sigma_eff, sigma_bad_mask=sigma_bad)

    # 4. Build PSF (pass sigma stamps as error maps)
    psf, _ = build_psf(sci_cuts, mask_list,
                        error_list=sig_cuts if sig_cuts else None,
                        oversampling=oversampling)
    psf_path = os.path.join(output_dir, psf_output_name)
    fits.writeto(psf_path, psf, overwrite=True)
    print(f"PSF saved: {psf_path}")

    # 5. Plot (via plot_psf_results)
    plot_psf_results(
        sci_image=sci_sub,
        catalog_file=catalog_file,
        psf_file=psf_path,
        valid_ids=valid_ids,
        sci_cutout_list=sci_cuts,
        mask_list=mask_list,
        valid_coords=valid_coords,
        output_dir=output_dir,
        plot_do_overview=plot_over,
        plot_stars_image=plot_stamps,
        plot_stars_scatter=plot_scatter,
        plot_stars_radial_profile=plot_radial,
        plot_psf_img=plot_psf_img,
        overview_name=overview_name,
        stamps_name=stamps_name,
        scatter_name=scatter_name,
        profile_name=profile_name,
        psfimg_name=psfimg_name,
        elongation_limit=elongation_limit,
        mag_bright_limit=mag_bright_limit,
        mag_faint_limit=mag_faint_limit,
        SNR_limit=SNR_limit,
        pixel_scale=pixel_scale,
    )

    return psf_path, valid_ids, sci_cuts, mask_list


# ═══════════════════════════════════════════════════════════════════
# Gaia utilities
# ═══════════════════════════════════════════════════════════════════

def gaia_crossmatch(catalog_file: str, max_distance: float = 0.5,
                    fetch_classprob: bool = True, backend: str = 'esa',
                    verbose: bool = True):
    """Crossmatch SEx catalog with Gaia DR3.

    Parameters
    ----------
    catalog_file : str
        Path to SEx catalog (must contain 'ra' and 'dec' columns).
    max_distance : float
        Crossmatch radius in arcsec.
    fetch_classprob : bool
        If True, query ESA Gaia for ``classprob_dsc_combmod_star``
        and merge into the result.
    backend : {'auto', 'cds', 'esa'}
        - ``'cds'`` — CDS XMatch service (``cdsxmatch.u-strasbg.fr``).
        - ``'esa'`` — ESA Gaia archive via ADQL + local k-d tree match.
        - ``'auto'`` — try CDS first, fall back to ESA on timeout (default).

    Returns
    -------
    df : pandas.DataFrame
    """
    import time
    import numpy as np
    import pandas as pd

    cat = ascii.read(catalog_file)
    ra_col  = 'ra'  if 'ra'  in cat.colnames else 'RA'
    dec_col = 'dec' if 'dec' in cat.colnames else 'DEC'

    if backend == 'auto':
        try:
            return _gaia_crossmatch_cds(cat, ra_col, dec_col,
                                        max_distance, fetch_classprob, verbose)
        except Exception as e:
            print(f"CDS XMatch failed ({e}), falling back to ESA…")
            return _gaia_crossmatch_esa(cat, ra_col, dec_col,
                                        max_distance, fetch_classprob, verbose)
    elif backend == 'cds':
        return _gaia_crossmatch_cds(cat, ra_col, dec_col,
                                    max_distance, fetch_classprob, verbose)
    elif backend == 'esa':
        return _gaia_crossmatch_esa(cat, ra_col, dec_col,
                                    max_distance, fetch_classprob, verbose)
    else:
        raise ValueError(f"backend must be 'auto', 'cds', or 'esa', got '{backend}'")


def _gaia_crossmatch_cds(cat, ra_col, dec_col, max_distance, fetch_classprob, verbose=True):
    """CDS XMatch backend."""
    from astroquery.xmatch import XMatch
    from astropy import units as u
    import time
    import numpy as np

    gaia = XMatch.query(
        cat1=cat, cat2='vizier:I/355/gaiadr3',
        max_distance=max_distance * u.arcsec,
        colRA1=ra_col, colDec1=dec_col
    )
    df = gaia.to_pandas()
    print(f"Gaia crossmatch (CDS): {len(df)}/{len(cat)} matched")

    if fetch_classprob and len(df) > 0:
        _fetch_classprob(df, verbose=verbose)
    return df


def _gaia_crossmatch_esa(cat, ra_col, dec_col, max_distance, fetch_classprob,
                         verbose=True,
                         max_retries=3):
    """ESA Gaia ADQL + local k-d tree backend (with retry)."""
    from astroquery.gaia import Gaia
    from scipy.spatial import cKDTree
    import time
    import numpy as np

    # 1. ADQL query for all Gaia sources in the field
    pad = 0.02  # deg
    ra_min, ra_max = cat[ra_col].min() - pad, cat[ra_col].max() + pad
    dec_min, dec_max = cat[dec_col].min() - pad, cat[dec_col].max() + pad

    query = f"""
    SELECT source_id, ra, dec, parallax, parallax_error,
           phot_g_mean_mag, bp_rp, ruwe, ipd_frac_multi_peak
    FROM gaiadr3.gaia_source
    WHERE ra BETWEEN {ra_min} AND {ra_max}
      AND dec BETWEEN {dec_min} AND {dec_max}
    """

    for attempt in range(1, max_retries + 1):
        try:
            job = Gaia.launch_job(query)
            gaia = job.get_results()
            break
        except Exception as e:
            if attempt < max_retries:
                wait = 10 * attempt
                print(f"  ESA query attempt {attempt}/{max_retries} failed "
                      f"({e}), retrying in {wait}s…")
                time.sleep(wait)
            else:
                raise
    print(f"Gaia ESA: {len(gaia)} sources in field")

    # 2. Local k-d tree crossmatch
    cat_coords = np.column_stack([cat[ra_col], cat[dec_col]])
    gaia_coords = np.column_stack([gaia['ra'], gaia['dec']])
    tree = cKDTree(gaia_coords)
    max_dist_deg = max_distance / 3600.0
    dist, idx = tree.query(cat_coords, distance_upper_bound=max_dist_deg)

    matched = dist < max_dist_deg
    print(f"Gaia crossmatch (ESA): {matched.sum()}/{len(cat)} matched")

    # 3. Build result DataFrame
    import pandas as pd
    rows = []
    for i in np.where(matched)[0]:
        g = gaia[idx[i]]; c = cat[i]
        rows.append({
            **{col: c[col] for col in cat.colnames},
            'angDist': dist[i] * 3600.,
            'Source': int(g['source_id']), 'RA_ICRS': float(g['ra']),
            'DE_ICRS': float(g['dec']), 'Plx': float(g['parallax']),
            'e_Plx': float(g['parallax_error']),
            'Gmag': float(g['phot_g_mean_mag']),
            'BP-RP': float(g['bp_rp']), 'RUWE': float(g['ruwe']),
            'IPDfmp': int(g['ipd_frac_multi_peak']),
        })
    df = pd.DataFrame(rows)

    if fetch_classprob and len(df) > 0:
        _fetch_classprob(df, verbose=verbose)
    return df


def _fetch_classprob(df, max_retries=3, verbose=True):
    """Fetch classprob_dsc_combmod_star from ESA Gaia and merge into df."""
    from astroquery.gaia import Gaia
    import time
    import numpy as np

    all_sids = sorted(set(int(s) for s in df['Source']))
    cp_map = {}
    n_batches = (len(all_sids) + 99) // 100
    n_failed = 0
    for b in range(0, len(all_sids), 100):
        batch = all_sids[b:b + 100]
        batch_id = b // 100
        for attempt in range(1, max_retries + 1):
            try:
                job = Gaia.launch_job(
                    f"SELECT source_id, classprob_dsc_combmod_star "
                    f"FROM gaiadr3.gaia_source "
                    f"WHERE source_id IN ({','.join(str(s) for s in batch)})")
                r = job.get_results()
                for row in r:
                    cp_map[int(row['source_id'])] = float(row['classprob_dsc_combmod_star'])
                break
            except Exception as e:
                if attempt < max_retries and verbose:
                    wait = 5 * attempt
                    print(f"  classprob batch {batch_id} attempt {attempt}/{max_retries} "
                          f"failed ({e}), retrying in {wait}s…")
                    time.sleep(wait)
                elif attempt >= max_retries and verbose:
                    print(f"  classprob batch {batch_id} failed after "
                          f"{max_retries} attempts: {e}")
                    n_failed += 1
        time.sleep(0.3)
    df['classprob_dsc_combmod_star'] = [
        cp_map.get(int(s), np.nan) for s in df['Source']]
    if verbose:
        n_found = sum(1 for v in df['classprob_dsc_combmod_star'] if not np.isnan(v))
        if n_failed > 0:
            print(f"  ESA classprob: {n_found}/{len(df)} ({n_failed}/{n_batches} batches failed)")
        else:
            print(f"  ESA classprob: {n_found}/{len(df)}")


def generate_background_mask(
    sci_image_file: str,
    seg_file: str,
    bkg_box_size: int = 64,
    bkg_filter_size: Tuple[int, int] = (3, 3),
    output_dir: str = "./"
    ):
    sci = fits.getdata(sci_image_file)
    seg = fits.getdata(seg_file)
    sci_smooth = gaussian_filter(sci, sigma=3.0)
    _, median, std = sigma_clipped_stats(sci_smooth, sigma=3.0)
    source_mask = (sci_smooth > median + 1.0 * std)
    source_mask = binary_dilation(source_mask, iterations=3)
    source_mask |= (seg > 0)  # Ensure all detected sources are masked
    bkg = Background2D(sci, (bkg_box_size, bkg_box_size), filter_size=bkg_filter_size,
                       mask=source_mask, bkg_estimator=MedianBackground(), sigma_clip=SigmaClip(sigma=3.0))
    sub = sci - bkg.background
    sub_smooth = gaussian_filter(sub, sigma=3.0)
    _, median, std = sigma_clipped_stats(sub_smooth, sigma=3.0)
    source_mask = (sub_smooth > median + 1.0 * std)
    source_mask = binary_dilation(source_mask, iterations=3)
    source_mask |= (seg > 0)  # Ensure all detected sources are masked
    hdu_prim = fits.PrimaryHDU(data=np.zeros((1,), dtype=np.uint8))
    hdu_bkg_mask = fits.ImageHDU(source_mask.astype(np.uint8), name='SOURCE_MASK')
    hdul = fits.HDUList([hdu_prim, hdu_bkg_mask])
    outfile = os.path.join(output_dir, 'background_mask.fits')
    hdul.writeto(outfile, overwrite=True)
    return source_mask, outfile