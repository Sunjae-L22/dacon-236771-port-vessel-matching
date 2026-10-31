#!/bin/sh
# author: opus
# Wait for builds 43-46, then 47 = 43+44 (D': dict/train v1+v2) and 48 = 45+46 (B: dict v1, train v1+v2).
R=/Users/sunjae/Desktop/항만; E=$R/shared/experiments
for n in 43 44 45 46; do
  until grep -q "zip size" $E/check_$n.log 2>/dev/null; do
    if grep -qE "Traceback|Killed|MemoryError" $E/make$n.log; then echo "build $n FAILED"; exit 1; fi
    sleep 60
  done
done
cd $R
python3 shared/src/merge_rankers.py submissions/47_submit.zip submissions/43_submit.zip submissions/44_submit.zip
python3 shared/src/merge_rankers.py submissions/48_submit.zip submissions/45_submit.zip submissions/46_submit.zip
python3 shared/src/local_check.py submissions/47_submit.zip > $E/check_47.log 2>&1
python3 shared/src/local_check.py submissions/48_submit.zip > $E/check_48.log 2>&1
python3 shared/src/zip_manifest.py $E/manifest.md submissions/31_submit.zip submissions/32_submit.zip submissions/33_submit.zip submissions/36_submit.zip submissions/42_submit.zip submissions/43_submit.zip submissions/44_submit.zip submissions/45_submit.zip submissions/46_submit.zip submissions/47_submit.zip submissions/48_submit.zip submissions/55_submit.zip
echo "47/48 ready $(date)"
