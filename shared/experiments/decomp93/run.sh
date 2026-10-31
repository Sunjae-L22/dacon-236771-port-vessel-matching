#!/bin/zsh
# author: opus — local_check for 79, 92, 93..96 with saved predictions
cd /Users/sunjae/Desktop/항만
for n in 79 93 94 95 96 92; do
  echo "== $n $(date +%T)"
  LOCAL_CHECK_SAVE=shared/experiments/decomp93/pred_$n.csv python3 shared/src/local_check.py submissions/${n}_submit.zip 2>&1 | tail -4
done
echo DONE
