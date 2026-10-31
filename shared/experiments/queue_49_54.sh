#!/bin/sh
# author: opus
# After 47/48 finish: v4-lite (49+50 -> 53) and C (51+52 -> 54), all KO_DICT=1, same seed pairs as 33.
R=/Users/sunjae/Desktop/항만; E=$R/shared/experiments; P=$E/pinned
until grep -q "zip size" $E/check_48.log 2>/dev/null; do sleep 60; done
launch() {  # num seed_offset pinned_dir dict_csv train_csv
  (cd $P/$3 && KO_DICT=1 LLM_DICT_CSV=$4 LLM_TRAIN_CSV=$5 SUBMIT_NUM=$1 SEEDS=3 SEED_OFFSET=$2 OUT_DIR=submit$1 \
    CACHE_DIR=$E/cache nohup sh -c "python3 make_submit.py && cd $R && python3 shared/src/add_exclusion.py submissions/$1_submit.zip submissions/$1_submit.zip && python3 shared/src/local_check.py submissions/$1_submit.zip > $E/check_$1.log 2>&1" > $E/make$1.log 2>&1 &)
}
V1=llm_korean_scored.csv; V12=llm_korean_scored_v12.csv
launch 49 0 build49 $V1 $V1
launch 50 5000 build49 $V1 $V1
launch 51 0 build43 $V12 $V1
launch 52 5000 build43 $V12 $V1
echo "launched 49-52 $(date)"
for n in 49 50 51 52; do
  until grep -q "zip size" $E/check_$n.log 2>/dev/null; do
    if grep -qE "Traceback|Killed|MemoryError" $E/make$n.log; then echo "build $n FAILED"; exit 1; fi
    sleep 60
  done
done
cd $R
python3 shared/src/merge_rankers.py submissions/53_submit.zip submissions/49_submit.zip submissions/50_submit.zip
python3 shared/src/merge_rankers.py submissions/54_submit.zip submissions/51_submit.zip submissions/52_submit.zip
python3 shared/src/local_check.py submissions/53_submit.zip > $E/check_53.log 2>&1
python3 shared/src/local_check.py submissions/54_submit.zip > $E/check_54.log 2>&1
python3 shared/src/zip_manifest.py $E/manifest.md submissions/3[1-6]_submit.zip submissions/4[2-8]_submit.zip submissions/49_submit.zip submissions/5[0-5]_submit.zip; echo "done $(date)"; tail -n 2 $E/check_49.log $E/check_50.log $E/check_51.log $E/check_52.log $E/check_53.log $E/check_54.log
