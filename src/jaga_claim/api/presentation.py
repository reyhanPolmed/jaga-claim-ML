"""Deterministic stakeholder wording derived from the existing scoring result.

No new findings, thresholds, causal attributions, or reason/evidence mappings.
"""
from .schemas import ScoreResponse, StakeholderExplanation

REASONS = {
    'COST_PEER_OUTLIER': (
        'Biaya lebih tinggi dibanding kasus sebanding',
        'Biaya tagih melewati batas pembanding untuk kelompok kasus serupa pada data historis.',
        'Periksa rincian tagihan dan dokumen pendukung bersama petugas klaim.'),
    'LOS_PEER_OUTLIER': (
        'Lama rawat lebih panjang dibanding kasus sebanding',
        'Jumlah hari perawatan melewati batas pembanding untuk kelompok kasus serupa.',
        'Konfirmasikan tanggal masuk, tanggal pulang, dan alasan lama perawatan pada dokumen layanan.'),
    'TEMPORAL_REPEAT': (
        'Pola kunjungan berdekatan perlu diperiksa',
        'Sistem menandai pengulangan atau tumpang tindih waktu layanan. Rincian pola tersedia pada bukti.',
        'Bandingkan tanggal dan dokumen episode layanan terkait untuk memahami apakah pengulangan memang sesuai kebutuhan.'),
    'POSSIBLE_DUPLICATE': (
        'Ada layanan yang sangat mirip',
        'Terdapat episode pembanding yang sangat mirip atau identik pada sejumlah atribut yang diperiksa.',
        'Bandingkan kedua episode dan dokumen layanan untuk memastikan apakah pelayanan berbeda atau terjadi pencatatan ganda.'),
    'CODING_MISMATCH': (
        'Data pengodean perlu dikonfirmasi',
        'Satu atau lebih aturan menandai kombinasi kode atau konteks layanan yang perlu dikonfirmasi.',
        'Minta petugas pengodean atau casemix memeriksa kode terhadap dokumen layanan dan aturan yang berlaku.'),
    'DATA_QUALITY': (
        'Kelengkapan atau konsistensi data perlu diperiksa',
        'Satu atau lebih aturan menandai data yang kosong, tidak valid, atau tidak konsisten.',
        'Cocokkan isian yang ditandai dengan sumber data; koreksi melalui sistem sumber bila diperlukan.'),
}
EVIDENCE_TITLES = {
    'billed_cost': 'Biaya yang ditagihkan',
    'los_days': 'Lama perawatan',
    'overlap_prior_inpatient': 'Waktu layanan yang tumpang tindih',
    'readmit_gap_days': 'Jarak waktu rawat inap ulang',
    'same_cbg_visits_30d': 'Kunjungan sebelumnya untuk kelompok kasus sama dalam tiga puluh hari',
    'dup_exact_ref': 'Episode pembanding dengan atribut identik',
    'dup_fuzzy_ref': 'Episode pembanding yang sangat mirip',
    'anomaly_score': 'Kombinasi karakteristik yang tidak lazim',
    'peer_features': 'Ketersediaan kelompok pembanding',
}


def stakeholder_explanation(result: ScoreResponse) -> StakeholderExplanation:
    labels = {'high': 'Tinggi', 'medium': 'Sedang', 'low': 'Rendah'}
    guidance = {
        'high': 'Klaim ini diprioritaskan untuk diperiksa lebih dahulu.',
        'medium': 'Klaim ini perlu ditinjau sesuai urutan dan kapasitas pemeriksaan.',
        'low': 'Klaim ini berada pada prioritas pemeriksaan rendah menurut model saat ini.',
    }
    reasons = [dict(code=code, title=REASONS[code][0], meaning=REASONS[code][1],
                    suggested_check=REASONS[code][2]) for code in result.reason_codes]
    if reasons:
        detail = ' Hal yang ditandai: ' + '; '.join(r['title'].lower() for r in reasons) + '.'
    elif any(e.feature == 'anomaly_score' for e in result.evidence):
        detail = ' Tidak ada kategori alasan khusus, tetapi terdapat bukti kombinasi karakteristik yang tidak lazim.'
    else:
        detail = ' Tidak ada kategori alasan khusus yang dikembalikan; hal ini tidak menjamin seluruh data klaim sudah benar.'
    return StakeholderExplanation(
        priority_label=labels[result.risk_band], summary=guidance[result.risk_band] + detail,
        score_meaning='Skor pada skala nol sampai satu membantu mengurutkan pemeriksaan. Skor yang lebih tinggi berarti prioritas lebih tinggi; nilainya bukan persentase kemungkinan fraud.',
        reasons=reasons,
        evidence=[dict(evidence_index=i, title=EVIDENCE_TITLES.get(e.feature,
                       'Temuan aturan pengodean' if e.feature.startswith('CM') else
                       'Temuan aturan kualitas data' if e.feature.startswith('DQ') else 'Bukti pendukung'),
                       observed_value=e.value, comparison=e.reference) for i, e in enumerate(result.evidence)],
        suggested_next_steps=['Baca alasan dan seluruh bukti pendukung bersama dokumen sumber.',
                              'Catat hasil pemeriksaan dan tindak lanjut oleh reviewer yang berwenang.'],
        limitations=['Kategori alasan merupakan indikator pemeriksaan, bukan bukti klaim salah atau keputusan penolakan.',
                     'Bukti berlaku untuk hasil secara keseluruhan; hubungan khusus setiap bukti dengan kategori alasan tidak ditetapkan.',
                     'Istilah pembanding: median adalah nilai tengah, kelompok sebanding adalah kasus dengan karakteristik serupa, dan persentil menunjukkan posisi relatif terhadap data pembanding.'])
