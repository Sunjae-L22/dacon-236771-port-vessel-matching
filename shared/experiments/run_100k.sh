#!/bin/sh
# author: opus — 72 recipe (build70 code, KO_DICT=1, aug seeds 0/1000/2000 + 5000/6000/7000) with the LLM
# Korean spellings swapped from Qwen v1 to Kanana-2-3B (spurious 제 rows dropped), dictionary and ranker rows both.
R=/Users/sunjae/Desktop/항만; E=$R/shared/experiments; P=$E/pinned/build100k
cd /tmp
for n in 900 901; do
  off=$(( (n - 900) * 5000 ))
  KO_DICT=1 LLM_DICT_CSV=llm_korean_scored_kanana_clean.csv LLM_TRAIN_CSV=llm_korean_scored_kanana_clean.csv \
    SUBMIT_NUM=$n SEEDS=3 SEED_OFFSET=$off OUT_DIR=submit$n python3 $P/make_submit.py > $E/make$n.log 2>&1 &
done
wait
mkdir -p $E/kanana_builds
for n in 900 901; do
  grep -q "wrote" $E/make$n.log || { echo "build $n FAILED"; exit 1; }
  mv $R/submissions/${n}_submit.zip $E/kanana_builds/
done
echo "100k builds done $(date)"
