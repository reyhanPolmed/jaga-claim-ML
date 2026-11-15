8 klaim sintetis untuk demo impor JAGA-CLAIM

File: synthetic_import_high_8_demo.jsonl
Schema: jaga-claim-curated-v1. UTF-8, satu objek per baris, revisi pertama.
Semua atribut/identitas adalah fiktif; ID baru tidak beririsan dengan file
demo sebelumnya. Tidak ada skor atau band yang dimasukkan dalam file.

Validasi 1 November 2026:
- 8 record lolos schema dan pemeriksaan field terlarang website.
- ScoringService hybrid-v1.1.0-dataset2025 pada prediction_time
  2026-11-01T00:00:00Z menghasilkan 8 high, 0 medium, 0 low.
- Baris 1-3: high / 0,45 / CODING_MISMATCH.
- Baris 4-5: high / 0,45 / DATA_QUALITY (tagihan nol disengaja).
- Baris 6-8: high / 0,6765 / COST_PEER_OUTLIER, CODING_MISMATCH,
  DATA_QUALITY (tagihan besar dan pengodean tidak selaras disengaja).

Impor melalui Data Klaim -> Import JSONL -> preview -> konfirmasi.
Pilih 8 klaim tersebut kemudian jalankan scoring dengan provider HTTP
ke model yang sesuai. Tidak perlu menghapus demo sebelumnya.
Impor ulang file identik akan dikenali sebagai snapshot duplikat.

Pengujian berupa validasi schema dan scoring langsung, belum impor database
atau klik browser. Model/konfigurasi lain dapat menghasilkan band berbeda.
Khusus demo, bukan data pasien nyata atau contoh keputusan klinis.
