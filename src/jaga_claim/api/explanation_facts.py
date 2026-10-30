"""Extract allowlisted facts, never forward free evidence text or identifiers."""
import re
from decimal import Decimal, InvalidOperation

from ..engines.rules import RULE_BY_ID
from .schemas import ScoreResponse

DRIVERS = {
    'log_billed': 'biaya tagih', 'cost_robust_z': 'perbedaan biaya terhadap kasus sebanding',
    'los_days': 'lama perawatan', 'los_robust_z': 'perbedaan lama perawatan terhadap kasus sebanding',
    'severity': 'tingkat keparahan', 'sec_dx_count': 'jumlah diagnosis tambahan',
    'sec_dx_chapter_count': 'keragaman kelompok diagnosis tambahan',
    'special_component_count': 'jumlah komponen tarif khusus', 'special_share': 'porsi tarif khusus',
    'procedure_count': 'jumlah prosedur', 'prior_visits_30d': 'jumlah kunjungan sebelumnya',
    'same_cbg_visits_30d': 'kunjungan sebelumnya pada kelompok kasus sama',
    'days_since_last_visit': 'jarak sejak kunjungan terakhir', 'age_years': 'usia',
    'tariff_recon_ratio': 'perbandingan tagihan dan tarif', 'is_inpatient': 'jenis rawat inap',
    'log_dx_cbg_rarity': 'kelangkaan kombinasi diagnosis dan kelompok kasus',
    'admit_primary_same': 'kesesuaian diagnosis masuk dan diagnosis utama',
}


def safe_facts(resp: ScoreResponse) -> list[dict]:
    # Import locally to avoid loading engine/model artefacts for template requests.
    from ..engine import ANOMALY_LABEL
    labels = {label: DRIVERS[key] for key, label in ANOMALY_LABEL.items() if key in DRIVERS}
    facts = []
    for e in resp.evidence[:50]:
        fact: dict = {}
        if e.feature == 'billed_cost':
            fact = {'type': 'cost_comparison'}
            if re.fullmatch(r'Rp\d{1,3}(?:\.\d{3})*|Rp\d+', e.value):
                fact['billed_cost'] = e.value
            for key, pattern in [('peer_median', r'Median peer (Rp[\d.]+);'),
                                 ('peer_p95', r'p95 peer (Rp[\d.]+);'),
                                 ('peer_count', r'\(n=(\d+)\)')]:
                match = re.search(pattern, e.reference)
                if match:
                    fact[key] = match[1]
            if len(fact) == 1:
                fact = {}
        elif e.feature == 'anomaly_score':
            fact = {'type': 'unusual_feature_combination'}
            pct = re.fullmatch(r'persentil (\d{1,3}(?:\.\d+)?)', e.value)
            if pct and 0 <= float(pct[1]) <= 100:
                fact['percentile'] = pct[1]
            drivers = re.search(r'pendorong utama: (.+)$', e.reference)
            if drivers:
                listed = drivers[1].split(', ')
                fact['review_indicators'] = [labels[s] for s in listed if s in labels][:3]
        elif e.feature in ('los_days', 'readmit_gap_days', 'same_cbg_visits_30d'):
            if re.fullmatch(r'\d{1,4}(?:\.\d+)? (?:hari|kunjungan sebelumnya)', e.value):
                fact = {'type': e.feature, 'observed': e.value}
        elif e.feature in ('dup_exact_ref', 'dup_fuzzy_ref'):
            fact = {'type': 'similar_episode', 'match': 'identical_checked_attributes' if e.feature == 'dup_exact_ref' else 'similar_checked_attributes'}
        elif e.feature == 'overlap_prior_inpatient':
            fact = {'type': 'overlapping_episode'}
        elif e.feature == 'peer_features':
            fact = {'type': 'insufficient_peer_data'}
        elif e.feature.startswith('rule:'):
            rule = RULE_BY_ID.get(e.feature.removeprefix('rule:'))
            if rule:
                # Registry-owned description, never the row's diagnosis/age/date/free text.
                fact = {'type': 'rule_finding', 'description': rule.description, 'status': 'DRAFT'}
        if fact:
            facts.append(fact)
    return facts


NUMBER = re.compile(r'(?<![\d.,])\d+(?:[.,]\d+)*')


def number_key(value: str) -> Decimal:
    # Indonesian grouped currency versus decimal scores/percentiles.
    if re.fullmatch(r'\d{1,3}(?:\.\d{3})+(?:,\d+)?', value):
        value = value.replace('.', '')
    try:
        return Decimal(value.replace(',', '.'))
    except InvalidOperation as exc:
        raise ValueError('unsupported numeric format') from exc


def numeric_values(value) -> set[Decimal]:
    if isinstance(value, dict):
        return set().union(*(numeric_values(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(numeric_values(v) for v in value)) if value else set()
    return {number_key(m.group()) for m in NUMBER.finditer(str(value))}
