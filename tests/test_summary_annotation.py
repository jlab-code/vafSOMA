"""annotate_summary_with_diagnostics marks unresolvable layers in the window summary without changing any count."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from vafsoma.estimate import annotate_summary_with_diagnostics


def test_annotation_marks_unresolvable_layers_and_keeps_counts():
    summ = pd.DataFrame({"tree": ["t1", "t2"], "mean_posterior_weighted_L1_count": [50.0, 10.0],
                         "mean_posterior_weighted_L2_count": [10.0, 20.0]})
    rv = pd.DataFrame({"tree": ["t1", "t1", "t2"], "layer": ["L1", "L2", "L2"],
                       "mean_model_implied_recall": [0.01, 0.93, 0.88], "resolvable": [False, True, True]})
    fit = pd.DataFrame({"tree": ["t1", "t2"], "fit_warning": [True, False]})
    out = annotate_summary_with_diagnostics(summ, rv, fit)
    pd.testing.assert_frame_equal(out[summ.columns], summ)            # counts untouched
    assert list(out["L1_status"]) == ["unresolvable", "no_diagnostics"]
    assert list(out["L2_status"]) == ["reportable", "reportable"]
    assert out.loc[0, "L1_resolvable"] == False and pd.isna(out.loc[1, "L1_resolvable"])
    assert np.isclose(out.loc[0, "L1_model_implied_recall"], 0.01) and np.isnan(out.loc[1, "L1_model_implied_recall"])
    assert list(out["fit_warning"]) == [True, False]
    assert list(out["L3_status"]) == ["no_diagnostics", "no_diagnostics"]


def test_annotation_without_diagnostics_is_a_no_op_on_counts():
    summ = pd.DataFrame({"tree": ["t1"], "mean_posterior_weighted_L2_count": [10.0]})
    out = annotate_summary_with_diagnostics(summ, pd.DataFrame(), None)
    assert out.loc[0, "L2_status"] == "no_diagnostics" and "fit_warning" not in out.columns


if __name__ == "__main__":
    test_annotation_marks_unresolvable_layers_and_keeps_counts()
    test_annotation_without_diagnostics_is_a_no_op_on_counts()
    print("ok")
