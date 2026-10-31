"""Patch pinned/build49 (copy of build43) into v4-lite: ch fix + LLM dictionary hygiene."""
import sys
from pathlib import Path

D = Path(sys.argv[1])
p = D / "ko_translit_argon.py"
s = p.read_text(encoding="utf-8")

# 1) ch fix: keep "ch" through the c->k rule with a placeholder
old = '    s = re.sub(r"tch|ch", "ch", s)\n    s = re.sub(r"sh(?=[iu])", "sy", s)'
assert s.count(old) == 1
s = s.replace(old, '    s = re.sub(r"tch|ch", "\\x01", s)  # placeholder: the c->k rule below used to turn every ch into kh\n'
                   '    s = re.sub(r"sh(?=[iu])", "sy", s)')
old = '    s = re.sub(r"q(?=[iu])", "ch", s)\n    s = re.sub(r"c(?=[eiy])", "s", s)\n    s = re.sub(r"c", "k", s)\n'
assert s.count(old) == 1
s = s.replace(old, '    s = re.sub(r"q(?=[iu])", "\\x01", s)\n    s = re.sub(r"c(?=[eiy])", "s", s)\n'
                   '    s = re.sub(r"c", "k", s)\n    s = s.replace("\\x01", "ch")\n')

# 2) LLM hygiene in _fit_train_pairs
a = s.index("    def _fit_train_pairs(self, train_pairs) -> None:")
b = s.index("            for kt, cnt in full_stem_counts.items():")
c = s.index("        self.exact_word_dict = word_dict")
head = s[a:b]
body = head[head.index("            pair_counts = defaultdict(Counter)"):]
helper = ('    def _count_pairs(self, records):\n'
          '        """Token and DP-subword pair counts plus whole-stem counts for one group of records."""\n'
          + "\n".join(l[4:] if l.startswith("    ") else l for l in body.rstrip("\n").split("\n")) + "\n"
          '        return pair_counts, full_stem_counts\n\n')
new_fit = '''    def _fit_train_pairs(self, train_pairs) -> None:
        """Learn token-level and DP-segmented subword Korean->English translations strictly from fold train data.

        Real train pairs are learned first and may override the universal dictionary. LLM-generated pairs
        (sample_id "LLM_*") only add keys nobody else defines: they never override the universal
        dictionary or a real pair (42: 글로리아->glory, 마린->marlin), never claim a whole Korean letter
        acronym (비에스->bw), skip 1-syllable keys and need a phonetic match.
        """
        base_dict = dict(_UNIVERSAL_KO_SUBWORDS)
        word_dict = dict(_UNIVERSAL_KO_SUBWORDS)
        if train_pairs is not None:
            if pd is not None and isinstance(train_pairs, pd.DataFrame):
                records = train_pairs.to_dict("records")
            else:
                records = list(train_pairs)
            is_llm = [str(r.get("sample_id") or "").startswith("LLM") for r in records]
            pair_counts, full_stem_counts = self._count_pairs([r for r, g in zip(records, is_llm) if not g])

            for kt, cnt in full_stem_counts.items():
                best_et, _ = cnt.most_common(1)[0]
                word_dict[kt] = best_et

            for kt, cnt in pair_counts.items():
                best_et, best_c = cnt.most_common(1)[0]
                if best_c >= 1:
                    if _pair_phon_sim(kt, best_et) >= 0.45 or best_c >= 2:
                        word_dict[kt] = best_et
                        if len(kt) >= 2:
                            base_dict[kt] = best_et

            protected = set(word_dict)
            llm_pairs, llm_stems = self._count_pairs([r for r, g in zip(records, is_llm) if g])
            for counts, is_stem in ((llm_stems, True), (llm_pairs, False)):
                for kt, cnt in sorted(counts.items()):
                    best_et, _ = cnt.most_common(1)[0]
                    acr = parse_ko_letter_acronym(kt)
                    if kt in protected or len(kt) < 2 or (acr is not None and not acr[1]):
                        continue
                    if _pair_phon_sim(kt, best_et) < 0.50:
                        continue
                    word_dict[kt] = best_et
                    if not is_stem:
                        base_dict[kt] = best_et

'''
s = s[:a] + helper + new_fit + s[c:]
p.write_text(s, encoding="utf-8")

# make_submit: keep sample_id in ko_pairs so the inference-time matcher can tell LLM rows apart
p = D / "make_submit.py"
m = p.read_text(encoding="utf-8")
old = 'llm_dict])[\n                 ["vessel_name", "target_vessel_id"]]'
assert old in m
m = m.replace(old, 'llm_dict])[\n                 ["sample_id", "vessel_name", "target_vessel_id"]]')
p.write_text(m, encoding="utf-8")
print("patched", D)
