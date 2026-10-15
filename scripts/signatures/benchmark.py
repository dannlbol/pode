#!/usr/bin/env python3
import argparse, json
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument("--features",type=int,required=True); p.add_argument("--folds",type=int,required=True)
p.add_argument("--min",dest="minimum",type=int,default=5); p.add_argument("--max",dest="maximum",type=int,default=60); p.add_argument("--output",default="benchmark.json")
a=p.parse_args(); counts=a.maximum-a.minimum+1
search=a.features+a.features*a.folds*counts
extraction=a.features*a.folds+a.features
result={"features":a.features,"folds":a.folds,"candidate_counts":counts,
"search_univariate_cox_fits":search,"fresh_extraction_univariate_cox_fits":extraction,
"total_univariate_cox_fits":search+extraction,
"note":"Literal legacy_exact workload; no cross-count or search-to-extraction fit reuse."}
Path(a.output).write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
