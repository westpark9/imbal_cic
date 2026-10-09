#!/usr/bin/env python3
"""Rebuild the 2026-10-07 manuscript tables from existing evidence; no inference."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/latex/20261007"
INPUTS = {}
TABLES = {}

def record(path):
    path = ROOT / path
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path

def read_json(path):
    return json.loads(record(path).read_text())

def fmt(x, digits=4):
    return f"{float(x):.{digits}f}"

def table(name, columns, body, caption, label, wide=False):
    env = "table*" if wide else "table"
    text = (
        f"\\begin{{{env}}}[t]\n\\centering\n"
        + f"\\caption{{{caption}}}\n\\label{{tab:{label}}}\n"
        + f"\\begin{{tabular}}{{{columns}}}\n\\toprule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"
        + f"\\end{{{env}}}\n"
    )
    p = OUT / "tables" / (name + ".tex")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    TABLES[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()

def row(items):
    return " & ".join(str(x) for x in items) + r" \\"

def main():
    sota = read_json("docs/research/20260930/exp61_report_data.json")
    expert = read_json("docs/research/20260930/exp63_capability_report_data.json")
    datasets = ("cic2018", "toniot")
    ds_names = {"cic2018": "CIC2018", "toniot": "ToN-IoT"}
    names = {
        "global_raw": "Global TabPFN-v3", "xgb": "XGBoost (100k)",
        "boostpfn": "BoostPFN", "localpfn": "LoCalPFN (FT)",
        "distpfn": "DistPFN (v3)", "xgb_full": "XGBoost (full)",
    }
    methods = list(names)
    clean_paths = [
        "data/derived/cic2018_conflict_free_fixed_split_20260915_180000/manifest.json",
        "data/derived/toniot_conflict_free_fixed_split_20260916_021730/manifest.json",
    ]
    clean = [read_json(p) for p in clean_paths]
    body = [row(["Population", "{CIC2018}", "{ToN-IoT}"]), r"\midrule"]
    for title, key in [("Original rows","rows_before"),("Removed rows","rows_removed"),("Retained rows","rows_after")]:
        body.append(row([title] + [d[key] for d in clean]))
    for split in ("train","val","test"):
        body.append(row([{"train":"Retained train","val":"Retained validation","test":"Retained test"}[split]] + [d["split_rows"][split] for d in clean]))
    table("quality", "@{}l *{2}{S[table-format=8.0]}@{}", body,
          "Offline conflict filtering. All rows of an identical-input group with inconsistent family labels are removed; retained rows keep their original split membership. Counts refer to full datasets and splits, not selected contexts.", "quality")

    body = [row(["Method", "{CIC2018}", "{ToN-IoT}"]), r"\midrule",
            r"\multicolumn{3}{l}{Same 100,000 training IDs} \\"]
    for m in methods:
        if m == "xgb_full":
            body += [r"\midrule", r"\multicolumn{3}{l}{Additional training information} \\"]
        values = [expert["datasets"][ds]["banks"]["4"]["models"]["full_test"]["global"]["summary"]["macro_f1"]
                  if m == "global_raw" else sota["datasets"][ds]["results"][m]["macro_f1"]
                  for ds in datasets]
        body.append(row([names[m]] + [fmt(value) for value in values]))
    body.append(row(["Proposed ($K=4$)"] + [fmt(expert["datasets"][ds]["banks"]["4"]["policy"]["macro_f1"]) for ds in datasets]))
    table("static", "@{}l *{2}{S[table-format=1.4]}@{}", body,
          "Static macro-F1 (seed 43). The first five methods share the Global training IDs. Full XGBoost uses the clean training pool; the proposed system also uses expert and route pools. These last two rows are not equal-information comparisons with the first five.", "static")

    labels = {"benign":"Benign","ddos":"DDoS","dos":"DoS","bot":"Bot","brute_force":"Brute force","web_attacks":"Web attack","infiltration":"Infiltration","mitm":"MitM","xss":"XSS"}
    body = [row(["Class", "{Test rows}", "{Global}", "{XGB 100k}", "{BoostPFN}", "{LoCalPFN}", "{DistPFN}", "{XGB full}", "{Proposed}"]), r"\midrule"]
    for ds in datasets:
        raw = sota["datasets"][ds]
        classmaps = {m:{v["name"]:v for v in raw["results"][m]["classes"]} for m in methods}
        # Use the exact paired Global outputs so a no-intervention policy has
        # identical per-class scores, rather than mixing a separately run Global.
        paired = expert["datasets"][ds]["banks"]["4"]["models"]["full_test"]["global"]["classes"]
        classmaps["global_raw"] = {v["class"]:v for v in paired}
        p = record(f"tabpfn/results/v4/exp63/20260930_exp63_k_sweep_s43/{ds}_k4/class_metrics.csv")
        with p.open() as f:
            policy = {r["class"]:r for r in csv.DictReader(f) if r["split"]=="full_test" and r["arm"]=="s1v0"}
        ordered = sorted(classmaps["global_raw"],key=lambda c:-classmaps["global_raw"][c]["support"])
        assert sum(classmaps["global_raw"][c]["support"] for c in ordered)==raw["test_rows"]
        body += [r"\multicolumn{9}{l}{"+ds_names[ds]+r"} \\"]
        for c in ordered:
            assert int(policy[c]["support"])==classmaps["global_raw"][c]["support"]
            body.append(row([labels.get(c,c.capitalize()),classmaps["global_raw"][c]["support"]] + [fmt(classmaps[m][c]["f1"]) for m in methods] + [fmt(policy[c]["system_f1"])]))
        if ds!=datasets[-1]:body.append(r"\midrule")
    table("classwise", "@{}l S[table-format=7.0] *{7}{S[table-format=1.4]}@{}", body,
          r"Per-class F1, sorted by descending test support within each dataset. Global and Proposed use paired predictions from the $K=4$ expert-policy evaluation; the remaining baselines use the same evaluation IDs. Proposed includes both scorer and verifier. Data-access qualifications from Table~\ref{tab:static} apply.", "classwise",True)

    body = [row(["$K$","{Macro-F1}","{Call (\\%)}","{Adopt (\\%)}","{Corrected}","{Damaged}"]),r"\midrule"]
    for ds in datasets:
        body.append(r"\multicolumn{6}{l}{"+ds_names[ds]+r"} \\")
        for k in ("2","4","6","8"):
            p=expert["datasets"][ds]["banks"][k]["policy"]
            body.append(row([k,fmt(p["macro_f1"]),fmt(100*p["proposed"]/p["rows"],2),fmt(100*p["accepted"]/p["rows"],2),p["helpful"],p["harmful"]]))
        if ds!=datasets[-1]:body.append(r"\midrule")
    table("policy","@{}r S[table-format=1.4] S[table-format=3.2] S[table-format=1.2] *{2}{S[table-format=5.0]}@{}",body,
          "Policy results for all tested expert counts (seed 43). Calls and adoptions are logical counts from cached predictions. Corrected/damaged counts compare with the paired Global prediction. Total context size differs across banks; no best $K$ is selected on test.", "policy")

    p=record("tabpfn/results/v7/exp70/20261002_exp70_class_arrival_100k_s43/summary.csv")
    with p.open() as f: arrivals=list(csv.DictReader(f))
    body=[row(["Dataset","Method","{Macro-F1}","{Benign FPR (\\%)}","{Update (s)}","{Infer (s)}"]),r"\midrule"]
    for ds in datasets:
        for m in ("xgb","tabpfn"):
            rows=[r for r in arrivals if r["dataset"]==ds and r["method"]==m]
            r=max(rows,key=lambda r:int(r["stage"]))
            assert int(r["context_rows"])==100000
            body.append(row([ds_names[ds],{"xgb":"XGBoost","tabpfn":"TabPFN"}[m],fmt(r["macro_f1"]),fmt(float(r["benign_false_alarm_rate"])*100,2),fmt(r["fit_seconds"],2),fmt(r["predict_seconds"],2)]))
    table("arrival","@{}ll S[table-format=1.4] S[table-format=2.2] S[table-format=2.2] S[table-format=3.2]@{}",body,
          "Final stage of the separate attack-introduction experiment (seed 43; 100,000 cumulative examples per dataset). No experts or S/V are used. Times are for final-stage updating and the complete final test batch; GPU jobs run sequentially. These are not cumulative times over all stages.", "arrival",True)

    body=[r" & \multicolumn{2}{c}{CIC2018} & \multicolumn{2}{c}{ToN-IoT} \\",
          row(["Method","{Fit (s)}","{Infer (s)}","{Fit (s)}","{Infer (s)}"]),r"\midrule"]
    for m in methods:
        vals=[]
        for ds in datasets:
            r=sota["datasets"][ds]["results"][m]
            vals += [fmt(r["fit_seconds"],1),fmt(r["predict_seconds"],1)]
        body.append(row([names[m]]+vals))
    table("cost","@{}l *{4}{S[table-format=5.1]}@{}",body,
          "Static-baseline elapsed times on a shared RTX 4090 (up to two concurrent workers). These include contention, not isolated latency. Global timing is from its baseline run, separate from the cached expert-policy evaluation. LoCalPFN includes validation in fit and retrieval in inference; DistPFN includes the shared Global computation. No online conditional-service latency is claimed for the proposed system.", "cost",True)

    record("docs/research/20261002/exp70_results.md")
    record("docs/research/20261002/exp70_protocol.md")
    record("docs/research/20260930/exp63_k_sweep_protocol.md")
    record("docs/research/20260929/exp61_local_run.md")
    record("docs/research/20260929/exp61_a100_run.md")
    for p in ["tabpfn/scripts/v4/exp62/exp62_sv_current_bank.py",
              "tabpfn/scripts/v4/exp63/exp63_k_sweep.py",
              "tabpfn/scripts/v4/exp51/nfv3_v3_exp51_scorer_target.py"]:
        record(p)
    for p in ["docs/latex/main.tex", "docs/latex/ref.bib", "docs/latex/acmart.cls",
              "docs/latex/ACM-Reference-Format.bst", "docs/latex/20261007/template/provenance.json",
              "scripts/common/build_manuscript_tables.py"]:
        record(p)
    sources={
        "date":"2026-10-07","status":"working_draft; single-seed exploratory evidence",
        "seed":43,"research_version_changed":False,
        "evidence":{"static_baselines":"v4/exp61","expert_policy":"v4/exp62 and v4/exp63","single_model_updates":"v7/exp70"},
        "rebuild":"python scripts/common/build_manuscript_tables.py",
        "inputs_sha256":INPUTS,"tables_sha256":TABLES,
        "literature_verified":{
            "hollmann2023tabpfn":"https://arxiv.org/abs/2207.01848",
            "hollmann2025tabpfn":"https://www.nature.com/articles/s41586-024-08328-6",
            "wang2025boostpfn":"https://proceedings.mlr.press/v258/wang25d.html",
            "thomas2024retrieval":"https://proceedings.neurips.cc/paper_files/paper/2024/hash/c40daf14d7a6469e65116507c21faeb7-Abstract-Conference.html",
            "lee2026distpfn":"https://arxiv.org/html/2605.04363v2",
            "wang2020long":"https://openreview.net/pdf?id=D9I3drBz4UC",
            "chen2016xgboost":"https://arxiv.org/abs/1603.02754",
            "luay2025NetFlowDatasetsV3":"https://arxiv.org/abs/2503.04404"},
        "submission_rules":"https://asiaccs2027.cityu.edu.mo/call-for-papers/index.html",
        "pending":["Final methodology and claims","Independent and repeated-seed evaluation","Equal-information and total-context controls","Measured online system cost","Anonymous artifact link and assigned publication metadata"],
    }
    (OUT/"sources.json").write_text(json.dumps(sources,indent=2)+"\n")
    print(f"Generated {len(TABLES)} tables from {len(INPUTS)} recorded inputs in {OUT}")

if __name__=="__main__":
    main()
