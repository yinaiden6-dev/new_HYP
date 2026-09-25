#!/usr/bin/env python3
"""Apply the same declared content-matching diagnostic to opened external panels.

No fitting, no model calls, no new data intake or candidate selection. This
tests transport of a statistical property, not external accuracy gains.
"""
import hashlib
from pathlib import Path
import numpy as np
from analyze_rc_mass_information_transport_v1 import ROOT, OUT, read, write, grouped, SOURCES, win


def main():
    assert not (OUT/'external_panels.json').exists()
    write(OUT/'external_panels_protocol.json',dict(
        declared='Before external panel computation; exactly the same free-content bands .005, .01, .02 as H593.',
        panels='All already-opened GroZi480 and ISIC537, original natural ColNomic C128.',
        quantity='Group-equal M concordance against all incorrect references with free-content difference within band; no rescue selection.',
        no_fit=True,no_new_forward=True,no_accuracy_computation=True))
    base = ROOT/'results/rc_simple_external_transfer_v1'
    pre = read(base/'preflight.json')
    outcomes = {}
    for name,authority_name in [('grozi','rc_new_hyp_grozi120_inference_authority_v1_20260913.json'),
                                ('isic','rc_new_hyp_isic_inference_authority_v1_20260914.json')]:
        authority = read(ROOT/'registry'/authority_name)
        if name == 'grozi':
            gallery = read(pre['legacy_gallery']['path'])['records']+read(authority['sources']['gallery_append']['path'])['records']
        else:
            gallery = read(authority['sources']['gallery']['path'])['records']
        identities = {g['physical_row']:g['identity'] for g in gallery}
        sealed = read(base/name/'result.json')
        rows = []
        for meta in sealed['rows']:
            folder = base/name/'queries'/meta['query_id']
            validation = read(folder/'validation.json')
            payload = Path(validation['payload']['path'])
            assert hashlib.sha256(payload.read_bytes()).hexdigest() == validation['payload']['sha256']
            d = read(payload)
            assert d['query_id'] == meta['query_id'] and len(d['axis']) == len(d['pairs']) == 128
            mask = np.array([identities[x] == meta['identity'] for x in d['axis']])
            assert bool(mask.any()) == meta['target_in_C128']
            if not mask.any():
                continue
            t, w = np.where(mask)[0], np.where(~mask)[0]
            l = np.array([p['free_content'] for p in d['pairs']]); m = np.array([p['mass'] for p in d['pairs']])
            r = dict(query_id=d['query_id'],component=meta['component'],raw_correct=meta['correct']['RAW'])
            for width in [.005,.01,.02]:
                values = []
                for i in t:
                    matched = w[abs(l[w]-l[i]) <= width]
                    if len(matched):
                        values.append(win(m[i],m[matched]))
                r['band_'+str(width)] = float(np.mean(values)) if values else None
            rows.append(r)
        assert len(rows) == sealed['target_recall_C128']
        outcomes[name] = dict(total=len(sealed['rows']),target_present=len(rows),group_unit=sealed['group_unit'],
            all_target_present={k:grouped(rows,k) for k in ['band_0.005','band_0.01','band_0.02']},
            RAW_wrong_target_present={k:grouped([r for r in rows if not r['raw_correct']],k) for k in ['band_0.005','band_0.01','band_0.02']})
        write(OUT/(name+'_per_query.json'),rows)
    write(OUT/'external_panels.json',dict(status='OPENED_EXTERNAL_INFORMATION_PROPERTY_RECOUNT_COMPLETE',
        panels=outcomes,sources=SOURCES,
        limitations='Content matching is observational. Previously used external panels, not fresh confirmation. ISIC is instance matching, not diagnosis. Source-video/patient bootstrap is exploratory and uncorrected.'))
    import json
    print(json.dumps(outcomes,indent=2))


if __name__ == '__main__':
    main()
