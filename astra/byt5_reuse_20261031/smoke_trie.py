from pathlib import Path
import sys,json,csv,time
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'package'))
from neural import setup,load_model,T5ForConditionalGeneration,T5Config
from trie_search import CatalogTrie,search,decode

def main():
    setup();tiny=T5ForConditionalGeneration(T5Config(vocab_size=384,d_model=32,d_ff=64,num_layers=1,num_decoder_layers=1,num_heads=2,d_kv=16,decoder_start_token_id=0,pad_token_id=0,eos_token_id=1,dropout_rate=0.0));tiny.eval()
    records=[dict(vessel_id=str(i),vessel_name=n) for i,n in enumerate(['A','AB','ALPHA','BETA','GAMMA','GAMMA'])];trie=CatalogTrie(records,{'0'})
    assert 'A' not in trie.by_name and trie.by_name['GAMMA']==['4','5']
    queries=['가나다','테스트'];before={q:search(tiny,q,trie,'cpu',2) for q in queries};after={q:search(tiny,q,trie,'cpu',2) for q in reversed(queries)};assert before==after
    # Real pretrained model and the actual allowed catalog validate the API and byte trie.
    catalog=list(csv.DictReader((ROOT/'package/data/vessel_catalog.csv').open(encoding='utf-8-sig')));excluded=set(json.loads((ROOT/'package/data/excluded_fold2.json').read_text()));full=CatalogTrie(catalog,excluded)
    model=load_model(Path('/tmp/astra_byt5_pretrained_path.txt').read_text(),'cpu');start=time.time();generated=search(model,'테스트',full,'cpu',16)
    assert generated and all(n['name'] in full.by_name and not (set(n['ids'])&excluded) for n in generated)
    report=dict(status='PASS_API_AND_TRIE_ONLY',tiny_order_mismatches=0,duplicate_names_preserved=True,excluded_ids_omitted=True,real_catalog_names=len(full.by_name),trie_nodes=full.nodes,pretrained_beam16_seconds=time.time()-start,generated_names=len(generated),fine_tuned_weights_tested=False,accuracy_tested=False)
    (ROOT/'TRIE_SMOKE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
