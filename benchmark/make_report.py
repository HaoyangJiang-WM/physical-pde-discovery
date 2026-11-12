from pathlib import Path
ROOT=Path(__file__).resolve().parent
text=(ROOT.parent/'docs/RESULTS.md').read_text(encoding='utf-8')
text=text.replace('(ROADMAP.md)','(../docs/ROADMAP.md)')
text=text.replace('[Original detailed report](../benchmark/RESULTS.md) · ','')
(ROOT/'RESULTS.md').write_text(text,encoding='utf-8')
print('Wrote English benchmark report')
