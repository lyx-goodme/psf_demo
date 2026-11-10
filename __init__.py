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
subtract_background()      — sky subtraction
generate_background_mask() — create source mask via SExtractor seg + dilation

Gaia
----
gaia_crossmatch()          — CDS XMatch or ESA ADQL + local k-d tree

GALFIT
------
write_psf_feedme()         — generate GALFIT config
run_galfit()               — execute GALFIT
extract_fitted_params()    — parse fitted parameters from log
plot_galfit_results()      — data / model / PSF / residual visualization

Plotting
--------
plot_psf_results()         — convenience: overview + stamps + scatter + profiles + PSF image
plot_overview()            — stars marked on full image
plot_star_stamps()         — grid of individual star cutouts
plot_scatter_diagnostics() — SNR-elongation and mag-flux_radius scatter
plot_radial_profiles()     — PSF + star median + percentile band 1D profiles
plot_psf_image()           — single PSF image with FWHM
plot_background()          — 2×2: original / mask / bkg model / subtracted
plot_segment()             — segmentation map quick look
"""

from .selection import (
    select_valid_stars,
    subtract_background,
    cutout_stars,
    build_psf,
    stacked_psf_pipeline,
    generate_background_mask,
    gaia_crossmatch,
)

from .galfit import (
    write_psf_feedme,
    run_galfit,
    extract_fitted_params,
    extract_chi2nu,
    plot_galfit_results,
)

from .plotting import (
    plot_psf_results,
    plot_overview,
    plot_star_stamps,
    plot_scatter_diagnostics,
    plot_psf_radial_profile,
    plot_radial_profiles,
    plot_psf_image,
    plot_background,
    plot_segment,
)