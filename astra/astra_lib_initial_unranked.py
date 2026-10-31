"""Author: Astra. Additive retrieval; all learned statistics use supplied fit rows."""
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz
import opus_lib as B

class Index(B.Index):
    def __init__(self, catalog, train=None):
        super().__init__(catalog, train)
        self.units = catalog.unit.fillna('').astype(str).to_numpy()
        self.size = catalog['size'].map(B.positive).to_numpy(float)
        self.meta = {}
        if train is not None:
            for field, values in [('flag_name', self.cflag), ('vessel_type', self.ctype), ('unit', self.units)]:
                source = 'vessel_type' if field == 'unit' else field
                # A repeatedly observed ship supplies one vote per observed label.
                sub = train.drop_duplicates(['target_vessel_id', source])
                counts = {}
                for row in sub.to_dict('records'):
                    i = self.id2i.get(row['target_vessel_id'])
                    if i is None: continue
                    key = str(row[source]); value = str(values[i])
                    counts.setdefault(key, {})[value] = counts.setdefault(key, {}).get(value, 0) + 1
                self.meta[field] = counts

    def confidence(self, field, key, values):
        cnt = self.meta.get(field, {}).get(str(key), {})
        total = sum(cnt.values())
        # Neutral when unsupported; small samples cannot produce near-zero certainty.
        return np.array([(cnt.get(str(v), 0) + 2.) / (total + 4.) for v in values], dtype=np.float32)

def extra_candidates(ix, row, limit=48):
    q = str(row.get('vessel_name') or '')
    gt, igt = [B.positive(row.get(k)) for k in ('gross_tonnage','international_gross_tonnage')]
    pool = set()
    for v in (gt, igt):
        if np.isfinite(v):
            lo, hi = np.searchsorted(ix.gt_sorted, [v / 1.4, v * 1.4])
            pool.update(ix.gt_sorted_idx[lo:hi].tolist())
    if not pool: return []
    pool = np.array(sorted(pool), dtype=int)
    if B.is_hangul(q):
        info = ix.ko.translate_ko(q)
        query = info['trans']
        names = [ix.ko.cat_en_compact[i] for i in pool]
        rom = info['rom_no_eu']
        projected = [ix.ko.cat_en2ko[i] for i in pool]
        scores = [process.cdist([query], names, scorer=s, workers=1)[0] for s in (fuzz.ratio, fuzz.partial_ratio)] if query else []
        if rom:
            scores += [process.cdist([rom], projected, scorer=s, workers=1)[0] for s in (fuzz.ratio, fuzz.partial_ratio)]
    else:
        query = B.norm(q)
        names = [ix.names[i] for i in pool]
        scores = [process.cdist([query], names, scorer=s, workers=1)[0] for s in (fuzz.ratio, fuzz.partial_ratio)] if query else []
    gap = np.minimum([B.rel_gap(gt, ix.gt[i]) for i in pool], [B.rel_gap(igt, ix.gt[i]) for i in pool])
    tp = ix.confidence('vessel_type', row.get('vessel_type'), ix.ctype[pool])
    fp = ix.confidence('flag_name', row.get('flag_name'), ix.cflag[pool])
    # Mild tie breakers, never discard a candidate solely because its tonnage differs.
    tie = -8. * gap + 1.5 * np.log(tp) + np.log(fp)
    out = set()
    for s in scores:
        order = np.lexsort((pool, -(s + tie)))[:limit]
        out.update(pool[order].tolist())
    meta = 4 * np.log(tp) + 2 * np.log(fp) - 30 * gap
    out.update(pool[np.lexsort((pool, -meta))[:24]].tolist())
    return sorted(out)

def query_candidates(ix, row):
    cand = B.query_candidates(ix, row)
    for i in extra_candidates(ix, row): cand.setdefault(i, [999, 999, 999])
    return dict(sorted(cand.items()))

EXTRA_FEATURES = ['flag_shrunk','type_shrunk','unit_shrunk','catalog_log_gt','catalog_log_dwt','catalog_log_size',
                  'dwt_gt_ratio','size_gt_ratio','gt_igt_gap','flag_support','type_support']
FEATURES = B.FEATURES + EXTRA_FEATURES

def query_features(ix, row, cand):
    idx, F = B.query_features(ix, row, cand)
    a = np.asarray(idx)
    gt, igt = [B.positive(row.get(k)) for k in ('gross_tonnage','international_gross_tonnage')]
    cols = [ix.confidence('flag_name', row.get('flag_name'), ix.cflag[a]),
            ix.confidence('vessel_type', row.get('vessel_type'), ix.ctype[a]),
            ix.confidence('unit', row.get('vessel_type'), ix.units[a]),
            np.log1p(ix.gt[a]), np.log1p(ix.dwt[a]), np.log1p(ix.size[a]),
            ix.dwt[a] / np.maximum(ix.gt[a],1), ix.size[a] / np.maximum(ix.gt[a],1),
            np.full(len(a), B.rel_gap(gt,igt)),
            np.full(len(a), sum(ix.meta.get('flag_name',{}).get(str(row.get('flag_name')),{}).values())),
            np.full(len(a), sum(ix.meta.get('vessel_type',{}).get(str(row.get('vessel_type')),{}).values()))]
    return idx, np.column_stack([F, *cols]).astype(np.float32)
