#!/usr/bin/env python3
"""Find code, aggregate results, and released models for an article figure or table."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def verify(rows):
    expected={f'Figure {i}' for i in range(1,7)}|{f'Table {i}' for i in range(1,4)}|{f'Figure S{i}' for i in range(1,9)}|{f'Table S{i}' for i in range(1,21)}
    if {r['item'] for r in rows}!=expected or len(rows)!=37:raise ValueError('The index must contain exactly the 37 article items')
    count=0
    for row in rows:
        if not row['scope'] or not (row['code'] or row['results']):raise ValueError('Entry needs resources and a scope description: '+row['item'])
        for kind in ['code','results','models']:
            for name in row[kind]:
                path=Path(name)
                if path.is_absolute() or '..' in path.parts or not (ROOT/path).is_file():raise ValueError('Invalid resource path: '+name)
                count+=1
    return {'status':'PASS','article_items':len(rows),'resource_links':count,'reported_tables':len(list((ROOT/'results/publication_tables').glob('Table_*.csv')))}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--item');p.add_argument('--verify',action='store_true');a=p.parse_args()
    rows=json.loads((ROOT/'paper_index.json').read_text())
    if a.verify:print(json.dumps(verify(rows),indent=2));return
    if a.item:
        match=[r for r in rows if r['item'].casefold()==a.item.casefold()]
        if not match:p.error('Use a figure/table name such as "Figure 5" or "Table S13"')
        print(json.dumps(match[0],indent=2,ensure_ascii=False));return
    for r in rows:print(r['item']+': '+r['title'])
if __name__=='__main__':main()
