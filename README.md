# psf_tools — DESI Legacy Survey Stacked PSF Pipeline

Build a stacked PSF from DESI Legacy Survey images via:
**SExtractor → morphology selection → Gaia crossmatch → cutout extraction → psfr stacking**.

Also includes GALFIT PSF fitting helpers for chi²ν quality assessment.

## Quick Start

```python
from psf_tools import stacked_psf_pipeline

psf_path, valid_ids, sci_cuts, mask_list = stacked_psf_pipeline(
    sci_image="image_r.fits",
    catalog_file="sex_out.cat",
    seg_file="sex_out_seg.fits",
    output_dir="./output",
    subtract_bkg=True,
    external_mask="background_mask.fits",
    sigma_image="sigma_r.fits",
)
```

See `psf_demo/demo_stacked_psf.ipynb` for a full walkthrough.

## Pipeline Overview

```
Raw Image → SExtractor → Source Mask → Background Subtraction
    → SExtractor (on subtracted) → select_valid_stars
    → Gaia Crossmatch (optional) → Cutout Stars → psfr → Stacked PSF
    → GALFIT Fitting → chi²ν Analysis
```

## Modules

| Module | Purpose |
|--------|---------|
| `selection.py` | Star selection, background subtraction, cutout extraction, PSF stacking, Gaia crossmatch |
| `galfit.py` | GALFIT feedme generation, execution, parameter extraction, diagnostic plots |
| `plotting.py` | Overview, stamps, radial profiles, scatter diagnostics, background comparison |

## Dependencies

- `numpy`, `pandas`, `scipy`, `matplotlib`
- `astropy`, `photutils`
- `psfr` — PSF stacking
- `galfitx` — SExtractor wrapper (for the demo notebook)
- `masktools` — mask expansion (for the demo notebook)
- `astroquery` — Gaia crossmatch (CDS XMatch / ESA ADQL)

## Demo Data

`psf_demo/demo_data/` contains a DESI Legacy Survey r-band cutout:

| File | Description |
|------|-------------|
| `image_r.fits` | Science image (nanomaggies) |
| `sigma_r.fits` | Per-pixel uncertainty |
| `wht_r.fits` | Weight map |
| `mask_r.fits` | Pre-computed mask |