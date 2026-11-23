#!/usr/bin/env python3
"""Recreate headline counts in manuscript Tables 2 and 3 from audited Stage3B JSON.

This is a saved-summary reproduction, not a new Monte Carlo run. All counts
refer to hypothetical parameter-design points, never sampled real firms.
"""
from pathlib import Path
import csv, json

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT/'stage3b/results/primary_findings.json').read_text())
RULES = ['constant_eta', 'reliance_weighted_eta']
OUT = ROOT/'output'
OUT.mkdir(exist_ok=True)

def p1(comp, threshold, rule, direction):
    rows = [r for r in DATA['P1_pooled']
            if r['comparison']==comp and r['threshold']==threshold
            and r['learning_rule']==rule and r['L']==1.0
            and r['access_direction']==direction]
    assert len(rows)==1, ('P1 missing or ambiguous',comp,threshold,rule,direction)
    return rows[0]

def tab2(comp,threshold,rule):
    if comp=='F100-B':
        rows=[p1(comp,threshold,rule,'increased')]
    else:
        rows=[p1(comp,threshold,rule,x) for x in ['increased','decreased']]
    return {'count':sum(r['dissociation_either'] for r in rows),
            'denominator':sum(r['design_points'] for r in rows),
            'confirmed':sum(r['confirmed_either'] for r in rows)}

tab2_rows=[]
for comp,t in [('F100-B',0.0),('F100-B',.02),('F100-B',.05),('C-B',.02),('D-Bplus',.02)]:
    row={'comparison':comp,'threshold':t,'source_label':'predecessor'}
    for rule in RULES:
        r=tab2(comp,t,rule)
        row[rule+'_count']=r['count'];row[rule+'_denominator']=r['denominator']
        row[rule+'_confirmed']=r['confirmed']
    tab2_rows.append(row)
assert [(r['constant_eta_count'],r['reliance_weighted_eta_count']) for r in tab2_rows]==[(163,162),(0,0),(0,0),(0,0),(1,1)]

tab3_rows=[]
for rule in RULES:
    matches=[r for r in DATA['checking_predecessor']
             if r['comparison']=='D-C' and r['learning_rule']==rule
             and r['outcome']=='post_shock__Q']
    assert len(matches)==1
    r=matches[0]
    tab3_rows.append({'rule':rule,'denominator':r['points'],
                      'positive_gt_002':r['positive'], 'positive_confirmed':r['confirmed_positive'],
                      'negative_lt_m002':r['negative'], 'negative_confirmed':r['confirmed_negative'],
                      'minimum':r['minimum'],'maximum':r['maximum']})
assert [(r['positive_gt_002'],r['negative_lt_m002']) for r in tab3_rows]==[(83,48),(79,64)]

for fname,rows in [('Table_2.csv',tab2_rows),('Table_3.csv',tab3_rows)]:
    with (OUT/fname).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('SIMULATED — Manuscript Table 2 (counts, numerator/denominator, confirmed)')
for r in tab2_rows:
    print(r['comparison'],r['threshold'],' | '.join(f'{rule}: {r[rule+"_count"]}/{r[rule+"_denominator"]} ({r[rule+"_confirmed"]} confirmed)' for rule in RULES))
print('SIMULATED — Manuscript Table 3 (checking D-C after change)')
for r in tab3_rows:
    print(r['rule'], '+',f'{r["positive_gt_002"]}/{r["denominator"]}', '-',f'{r["negative_lt_m002"]}/{r["denominator"]}')
print('Results written to',OUT)

if __name__=='__main__':
    pass
