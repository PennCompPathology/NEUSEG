"""
One-call evaluation pipeline for a single slide.

evaluate_slide(curr_slide_name, eval_cohort_df) ties the per-slide steps together:
  get_neuseg_results -> get_manual_results (reusing neuseg_res) ->
  annotation_to_contour_distances for both the NEUSEG and manual annotations.
Contours + mpp are computed once (in get_neuseg_results) and reused by the rest.
"""

from Helper_Functions.loading import get_neuseg_results, get_manual_results
from Helper_Functions.metrics import annotation_to_contour_distances


def evaluate_slide(curr_slide_name, eval_cohort_df, verbose=False):
    """
    Run the full annotation-to-contour evaluation for one slide.

    Parameters
    ----------
    curr_slide_name : str        slide Filename in eval_cohort_df (no .svs)
    eval_cohort_df : DataFrame    cohort table; looked up by 'Filename', must have
                                  neuseg_path / WSI_path / manual_path columns
    verbose : bool                pass-through to every step; when True shows the
                                  NEUSEG & manual overlays, the per-boundary
                                  validation figures, and prints the macro-means

    Returns
    -------
    dict with keys:
      row            : the eval_cohort_df row for this slide
      neuseg_res     : get_neuseg_results output (contour_gm_wm/csf, mpp, annot, ...)
      man            : get_manual_results output (manual annot, ...)
      neuseg_dist_df : NEUSEG annotation -> NEUSEG contour distances (per roi/boundary)
      manual_dist_df : manual annotation -> NEUSEG contour distances (per roi/boundary)
    """
    row = eval_cohort_df.set_index("Filename").loc[curr_slide_name]

    # NEUSEG first: contours + mpp + annotation computed once, reused below.
    neuseg_res = get_neuseg_results(curr_slide_name, row["neuseg_path"], row["WSI_path"], verbose=verbose)
    # Manual next: reuses neuseg_res (its SVS + contours) -- no recompute.
    man = get_manual_results(curr_slide_name, row["manual_path"], neuseg_res=neuseg_res, verbose=verbose)

    # Annotation-to-contour distances (both reuse neuseg_res).
    neuseg_dist_df = annotation_to_contour_distances(neuseg_res["annot"], neuseg_res, verbose=False)
    manual_dist_df = annotation_to_contour_distances(man["annot"], neuseg_res, verbose=verbose)

    if verbose:
        for name, dist_df in [("NEUSEG", neuseg_dist_df), ("manual", manual_dist_df)]:
            print(f"{name} per-boundary macro-mean (um):",
                  dist_df.groupby("boundary")["mean_um"].mean().round(3).to_dict())

    return {
        "row":            row,
        "neuseg_res":     neuseg_res,
        "man":            man,
        "neuseg_dist_df": neuseg_dist_df,
        "manual_dist_df": manual_dist_df,
    }
