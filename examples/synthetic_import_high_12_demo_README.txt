12 klaim sintetis berprioritas tinggi untuk demo JAGA-CLAIM

File: synthetic_import_high_12_demo.jsonl
Schema jaga-claim-curated-v1, UTF-8, satu objek per baris, revisi 1.
Semua identitas dan atribut fiktif. ID baru tidak beririsan dengan file demo
sebelumnya. Skor dan band tidak disertakan dalam file impor.

Verifikasi 1 November 2026:
- 12 record lolos curatedRowSchema dan detectForbiddenFields website.
- Claim ID dan participant key unik serta berbeda dari fixture sebelumnya.
- ScoringService model hybrid-v1.1.0-dataset2025 dengan prediction_time
  2026-11-01T00:00:00Z menghasilkan 12 high, 0 medium, 0 low.
- Baris 1-4: skor 0,45, CODING_MISMATCH (ketidaksesuaian pengodean disengaja).
- Baris 5-8: skor 0,45, DATA_QUALITY (biaya tagih nol disengaja).
- Baris 9-12: skor 0,6765, COST_PEER_OUTLIER, CODING_MISMATCH,
  DATA_QUALITY (biaya tinggi terhadap pembanding/tarif dan pengodean tidak selaras).

Login admin -> Data Klaim -> Import JSONL -> preview -> konfirmasi.
Pilih 12 klaim lalu jalankan scoring dengan SCORING_PROVIDER=http ke model
yang sama. Tidak perlu menghapus data demo sebelumnya.

Validasi dilakukan pada schema dan layanan scoring langsung, bukan melalui
klik browser atau penyimpanan database. Impor identik kembali menjadi snapshot
duplikat. Model/konfigurasi lain dapat mengubah skor/band. Khusus demo/test,
bukan data pasien nyata atau contoh keputusan klinis.
