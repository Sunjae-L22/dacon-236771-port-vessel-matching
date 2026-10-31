#!/bin/sh
# author: opus
# Same pipeline as merge_47_48.sh + queue_49_54.sh, but every python runs from /tmp with absolute paths:
# since 10/27 17:26 a python process whose working directory is inside ~/Desktop hangs at startup.
R=/Users/sunjae/Desktop/항만; E=$R/shared/experiments; P=$E/pinned; S=$R/submissions
cd /tmp
excl_check() {  # num: post-hoc exclusion (if not applied yet) + local check
  unzip -p $S/$1_submit.zip script.py | grep -q "Evaluation ships do not appear" || python3 $R/shared/src/add_exclusion.py $S/$1_submit.zip $S/$1_submit.zip
  python3 $R/shared/src/local_check.py $S/$1_submit.zip > $E/check_$1.log 2>&1
}
wait_zip() {  # num: wait until make_submit wrote the zip, then stop the wrapper's hung follow-up step
  until grep -q "wrote .*$1_submit.zip" $E/make$1.log 2>/dev/null; do
    if grep -qE "^(Traceback|MemoryError)|Killed" $E/make$1.log && ! grep -q "wrote " $E/make$1.log; then echo "build $1 FAILED"; exit 1; fi
    sleep 60
  done
  sleep 5; pkill -f "add_exclusion.py submissions/$1_submit.zip"; pkill -f "local_check.py submissions/$1_submit.zip"
}
for n in 43 44; do excl_check $n; done
for n in 45 46; do wait_zip $n; excl_check $n; done
python3 $R/shared/src/merge_rankers.py $S/47_submit.zip $S/43_submit.zip $S/44_submit.zip
python3 $R/shared/src/merge_rankers.py $S/48_submit.zip $S/45_submit.zip $S/46_submit.zip
python3 $R/shared/src/local_check.py $S/47_submit.zip > $E/check_47.log 2>&1
python3 $R/shared/src/local_check.py $S/48_submit.zip > $E/check_48.log 2>&1
echo "47/48 ready $(date)"
launch() {  # num seed_offset pinned_dir dict_csv train_csv
  KO_DICT=1 LLM_DICT_CSV=$4 LLM_TRAIN_CSV=$5 SUBMIT_NUM=$1 SEEDS=3 SEED_OFFSET=$2 OUT_DIR=submit$1 CACHE_DIR=$E/cache \
    nohup python3 $P/$3/make_submit.py > $E/make$1.log 2>&1 &
}
V1=llm_korean_scored.csv; V12=llm_korean_scored_v12.csv
launch 49 0 build49 $V1 $V1; launch 50 5000 build49 $V1 $V1; launch 51 0 build43 $V12 $V1; launch 52 5000 build43 $V12 $V1
echo "launched 49-52 $(date)"
for n in 49 50 51 52; do wait_zip $n; excl_check $n; done
python3 $R/shared/src/merge_rankers.py $S/53_submit.zip $S/49_submit.zip $S/50_submit.zip
python3 $R/shared/src/merge_rankers.py $S/54_submit.zip $S/51_submit.zip $S/52_submit.zip
python3 $R/shared/src/local_check.py $S/53_submit.zip > $E/check_53.log 2>&1
python3 $R/shared/src/local_check.py $S/54_submit.zip > $E/check_54.log 2>&1
python3 $R/shared/src/zip_manifest.py $E/manifest.md $S/3[1-6]_submit.zip $S/4[2-9]_submit.zip $S/5[0-5]_submit.zip
echo "done $(date)"
