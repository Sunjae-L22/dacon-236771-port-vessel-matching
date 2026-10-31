"""Unfiltered permitted-v1 LLM aliases on outerfold2; never use global score field."""
from pathlib import Path
import sys,json,hashlib,re,unicodedata,collections
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'shared/src'));import cv
from build_benchmark import representative,QUERY_COLUMNS,is_ko,SEED

def alpha_tokens(name):return set(re.findall('[A-Z]+',str(name).upper()))

def main():
    tr,cat=cv.load();folds=cv.folds(tr);fit=tr[folds!=2];held=tr[folds==2];fit_ids=set(fit.target_vessel_id);held_ids=set(held.target_vessel_id)
    source=ROOT/'shared/experiments/pinned/build70/llm_korean_scored.csv'
    # Deliberately do not read the globally computed score column at all.
    raw=pd.read_csv(source,usecols=['vessel_id','ko','src']);raw['source_llm_row']=np.arange(len(raw));raw=raw[raw.vessel_id.isin(held_ids)].copy();raw_count=len(raw)
    raw=raw[raw.ko.notna()].copy();raw['ko']=raw.ko.map(lambda s:unicodedata.normalize('NFC',str(s)).strip());raw=raw[raw.ko.map(is_ko)&raw.ko.str.len().gt(0)];syntactic_count=len(raw)
    raw=raw.drop_duplicates(['vessel_id','ko'],keep='first');metadata={r['target_vessel_id']:r for r in representative(held)};catalog_names=dict(zip(cat.vessel_id,cat.vessel_name));vocabulary=set().union(*(alpha_tokens(name) for name in cat[cat.vessel_id.isin(fit_ids)].vessel_name))
    rows=[]
    for r in raw.to_dict('records'):
        target=r['vessel_id'];base=metadata[target];row={k:base[k] for k in QUERY_COLUMNS};row['vessel_name']=r['ko'];row['call_sign']='UNSEEN';row['target_vessel_id']=target
        query_hash=hashlib.sha256((target+'\0'+r['ko']).encode()).hexdigest()[:20];tokens=alpha_tokens(catalog_names[target]);seen=tokens&vocabulary
        coverage='no_alpha_tokens' if not tokens else 'all_tokens_seen' if tokens<=vocabulary else 'some_tokens_seen' if seen else 'no_tokens_seen'
        row.update(query_id=f"B97_LLM_F2::{r['src']}::{target}::{query_hash}",source_sample_id=base['sample_id'],source_llm_row=int(r['source_llm_row']),source_language='ko',native_metadata_source_language='ko' if is_ko(base['vessel_name']) else 'en',original_vessel_name=base['vessel_name'],variant='llm_alias_'+str(r['src']),family='llm_natural_transliteration',cohort='llm_outer2',generation_source=str(r['src']),fit_token_coverage=coverage,alpha_token_count=len(tokens),seen_alpha_token_count=len(seen),name_origin='permitted_train_only_llm_v1',metadata_origin='actual_outerfold2_train_row',label_status='synthetic_alias_assumed_target_not_manually_verified',priority='separate_llm_anchor')
        rows.append(row)
    frame=pd.DataFrame(rows);counts=frame.groupby('target_vessel_id').size();frame['evaluation_weight']=frame.target_vessel_id.map(lambda target:1/float(counts[target]));assert frame.query_id.is_unique;assert not set(frame.target_vessel_id)&fit_ids;assert set(frame.target_vessel_id)<=held_ids
    for row in frame.to_dict('records'):
        base=metadata[row['target_vessel_id']]
        for key in ['gross_tonnage','international_gross_tonnage','flag_name','vessel_type','arrival_year']:
            assert (pd.isna(row[key]) and pd.isna(base[key])) or row[key]==base[key]
    target=OUT/'llm_holdout_fold2.csv';frame.to_csv(target,index=False)
    coverage=frame.groupby(['generation_source','fit_token_coverage']).agg(rows=('query_id','size'),vessels=('target_vessel_id','nunique')).reset_index();coverage.to_csv(OUT/'llm_holdout_coverage.csv',index=False)
    report=dict(status='READY_FOR_MATCHED_BASELINE_EVALUATION',outer_fold=2,representative_seed=SEED,raw_outerfold_rows=raw_count,syntactically_valid_rows=syntactic_count,unique_alias_pairs=len(frame),unique_vessels=frame.target_vessel_id.nunique(),fit_target_overlap=0,global_score_column_read=False,score_filter_applied=False,quality_filter='Only nonempty NFC-normalized Korean strings; exact (vessel_id,alias) deduplication',source_counts=frame.generation_source.value_counts().to_dict(),token_coverage_vessels=frame.drop_duplicates('target_vessel_id').fit_token_coverage.value_counts().to_dict(),allowed_query_columns=QUERY_COLUMNS,query_key='query_id',metadata_copy_verification='All numeric/category/year fields exactly match the selected real heldout row; call_sign intentionally replaced with UNSEEN',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),csv_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),metric_contract=['Overall headline is macro accuracy across target vessels, with aliases averaged within each vessel.', 'Also report by generation_source and fit_token_coverage, re-averaging within vessels in each subgroup.', 'Report native Korean, LLM aliases, and partial-name stress as separate cohorts; do not pool into one score.', 'Report full candidate recall, top20/top100 recall, and ranker accuracy independently.', 'These are assumed synthetic aliases generated from permitted train-only inputs, not manually verified native Korean labels and not a Public-LB proxy.', 'No extra model fitting or tuning is permitted on these heldout aliases. The original global score column is not read or used.'])
    (OUT/'llm_holdout_protocol.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
