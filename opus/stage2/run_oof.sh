#!/bin/zsh
# author: opus — stage-1 OOF for folds 0,1 then 3,4 (two at a time; four at once hit exit137 on 10/29)
cd /tmp
S=/Users/sunjae/Desktop/항만/opus/stage2
for pair in "0 1" "3 4"; do
  for f in ${=pair}; do python3 $S/oof.py --fold $f > $S/oof_fold$f.log 2>&1 & done
  wait
done
python3 $S/summarize.py
