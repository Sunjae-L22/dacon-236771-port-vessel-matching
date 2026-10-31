#!/bin/sh
# author: opus
# 70/71 = 33 recipe (build43 code, KO_DICT=1, LLM v1 dict/train, offsets 0/5000) + wide tonnage noise on
# 30% of the name-corrupted copies (pinned/build70). 72 = 70+71, compared with 33 (same seed pair).
# Python runs from /tmp with absolute paths (a cwd inside ~/Desktop hung python on 10/27).
R=/Users/sunjae/Desktop/항만; E=$R/shared/experiments; S=$R/submissions
cd /tmp
for n in 70 71; do
  off=$(( (n - 70) * 5000 ))
  KO_DICT=1 LLM_DICT_CSV=llm_korean_scored.csv LLM_TRAIN_CSV=llm_korean_scored.csv SUBMIT_NUM=$n SEEDS=3 \
    SEED_OFFSET=$off OUT_DIR=submit$n CACHE_DIR=$E/cache nohup python3 $E/pinned/build70/make_submit.py > $E/make$n.log 2>&1 &
done
wait
for n in 70 71; do
  grep -q "wrote .*${n}_submit.zip" $E/make$n.log || { echo "build $n FAILED"; exit 1; }
  python3 $R/shared/src/add_exclusion.py $S/${n}_submit.zip $S/${n}_submit.zip
done
python3 $R/shared/src/merge_rankers.py $S/72_submit.zip $S/70_submit.zip $S/71_submit.zip
python3 $R/shared/src/local_check.py $S/72_submit.zip > $E/check_72.log 2>&1
python3 $R/shared/src/zip_manifest.py $E/manifest_72.md $S/33_submit.zip $S/72_submit.zip
echo "72 ready $(date)"; tail -3 $E/check_72.log
