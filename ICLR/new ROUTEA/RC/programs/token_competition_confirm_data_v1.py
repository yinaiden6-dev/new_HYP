#!/usr/bin/env python3
"""Confirmation-only adapter: V2 evidence receipt remains bound to V2 origin."""
from token_competition_confirm_common_v1 import bind, checked, read, verify_confirmation
import token_competition_data_v2 as origin_data

ARMS = origin_data.ARMS
verify_sources = origin_data.verify_sources


def verify_panel(protocol):
    origin = verify_confirmation(protocol)
    # The original nested partitions and original seed0 contract are verified
    # directly; the confirmation's seed registry is independently checked above.
    return origin_data.verify_panel(origin)


def verify_evidence_gate(protocol):
    origin = verify_confirmation(protocol)
    original_receipt = origin_data.verify_evidence_gate(origin)
    if original_receipt != protocol['origin_evidence_validation']:
        raise ValueError('Original V2 evidence receipt changed')
    receipt = read(checked(original_receipt))
    if receipt['protocol'] != protocol['origin_protocol']:
        raise ValueError('Evidence receipt must bind the ORIGINAL V2 protocol')
    if receipt['manifest'] != bind(origin['evidence_manifest']):
        raise ValueError('Original V2 evidence manifest changed')
    # common_features, gallery, expected_records, panel and all folds are among
    # immutable_origin_fields; they bind physical C128 axes and F128 identities.
    return original_receipt


class TokenInputs(origin_data.TokenInputs):
    def __init__(self, protocol, fold, arm, pilot_query_id=None):
        if pilot_query_id is not None:
            raise ValueError('Confirmation reuses the validated V2 engineering pilot; no new pilot scope')
        origin = verify_confirmation(protocol)
        verify_evidence_gate(protocol)
        if fold not in list(origin['folds'].values()):
            raise ValueError('Confirmation split is not an original V2 split')
        super().__init__(origin, fold, arm)
        self.protocol = protocol
