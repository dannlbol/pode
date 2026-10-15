import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import prognostk as pk
from conftest import make_toy, toy_cfg


def test_palette_and_pvalue():
    cfg = toy_cfg()
    pal = pk.prognostk_palette(cfg, "risk", cfg["risk_groups"]["labels"])
    assert len(pal) == len(cfg["risk_groups"]["labels"])
    # unmapped level gets an auto colour
    assert len(pk.prognostk_palette(cfg, "Sex", ["M", "F", "Unknown"])) == 3
    assert pk.format_pvalue(0.0001).startswith("p")


def test_plots_write_files(tmp_path):
    cfg = toy_cfg(figure={"outdir": str(tmp_path)})
    d = make_toy()
    fit = pk.fit_cox(d, cfg)
    d = d.assign(risk=pk.assign_risk_groups(pk.predict_survival_at_time(fit, d, cfg=cfg), cfg))

    pk.plot_km(d, cfg, group_col="risk", filename="km.pdf")
    pk.plot_distribution(d, cfg, group_col="grp", kind="violin", filename="viol.pdf")
    pk.plot_distribution(d, cfg, group_col="grp", kind="box", filename="box.pdf")
    pk.plot_forest({"a": d.iloc[:75], "b": d.iloc[75:]}, cfg, filename="forest.pdf")
    pk.plot_calibration(d, cfg, pred=pk.predict_survival_at_time(fit, d, cfg=cfg),
                        n_bins=4, filename="calib.pdf")
    pk.plot_individual_curves(fit, d, cfg, filename="indiv.pdf")
    for f in ["km.pdf", "viol.pdf", "box.pdf", "forest.pdf", "calib.pdf", "indiv.pdf"]:
        assert os.path.exists(tmp_path / f)


def test_oncoplot_writes(tmp_path):
    cfg = toy_cfg(figure={"outdir": str(tmp_path)},
                  oncoplot={"tracks": [
                      {"name": "Sex", "column": "Sex", "type": "categorical"},
                      {"name": "Score", "column": "metaCpG", "type": "line"}]})
    pk.plot_oncoplot(make_toy(), cfg, filename="onco.pdf")
    assert os.path.exists(tmp_path / "onco.pdf")
