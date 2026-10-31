"""Decode a ten-decimal diagnostic Public score with English fixed at 3506/3568.

Usage: python decode_score.py 0.7829448903
Only returns Korean correct count; never estimates cohort size from OOF precision.
"""
from decimal import Decimal, localcontext
import argparse
import json

def decode(score):
    with localcontext() as ctx:
        ctx.prec=40
        value=Decimal(score)
        tolerance=Decimal('0.00000000005')
        english=Decimal('0.3')*3506/3568
        matches=[]
        for k in range(533):
            expected=english+Decimal('0.7')*k/532
            if abs(expected-value)<=tolerance:
                matches.append(dict(korean_correct=k,english_correct=3506,expected_score=str(expected)))
    if len(matches)!=1:
        raise ValueError('Not a unique solution: verify ten-decimal score, English invariance and denominators.')
    return matches[0]

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('score',help='Score reported to ten decimal places')
    args=p.parse_args()
    print(json.dumps(decode(args.score),indent=2))
    print('Mask probe: Korean correct count only. Cohort size/error count not identified; overlapping probes must not be summed.')
