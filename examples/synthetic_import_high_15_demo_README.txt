15 klaim sintetis berprioritas tinggi untuk demo JAGA-CLAIM

File impor: synthetic_import_high_15_demo.jsonl
Schema: jaga-claim-curated-v1, UTF-8, satu objek JSON per baris.
Semua ID dan atribut merupakan data fiktif. ID memakai namespace demo baru
yang berbeda dari fixture 10 klaim sebelumnya; source_revision semuanya 1.
Tidak ada risk_score/risk_band, identifier asli, atau field pascaverifikasi
yang ditulis dalam file impor. Hasil prioritas ditentukan oleh model.

Verifikasi 1 November 2026:
- 15 baris lolos curatedRowSchema dan detectForbiddenFields website.
- 15 claim_id dan participant_key masing-masing unik.
- ScoringService model hybrid-v1.1.0-dataset2025, prediction_time
  2026-11-01T00:00:00Z menghasilkan 15 high, nol medium, nol low.

Rincian skenario:
1-5: high / 0,45 / CODING_MISMATCH.
     Diagnosis spesifik jenis kelamin sengaja tidak selaras dengan peserta.
6-10: high / 0,45 / DATA_QUALITY.
      Nilai biaya tagih nol sengaja menguji aturan kualitas data penting.
11-15: high / 0,6765 / COST_PEER_OUTLIER, CODING_MISMATCH, DATA_QUALITY.
       Biaya tinggi terhadap pembanding/tarif dan ketidaksesuaian pengodean
       sengaja digabung agar demo memperlihatkan beberapa jenis bukti.

Penggunaan:
Login admin -> Data Klaim -> Import JSONL -> preview -> konfirmasi.
Pilih 15 klaim lalu jalankan scoring dengan SCORING_PROVIDER=http ke model
lokal yang sama. Buka Risk Queue/AI Analysis dan Reviewer Workspace.
Impor saja tidak membuat hasil scoring. File yang diimpor ulang tanpa
perubahan akan dikenali sebagai snapshot duplikat.

Pengujian di atas berupa validasi schema dan scoring langsung; belum
melakukan klik browser atau memasukkan data ke database website. Hasil bisa
berbeda bila model, aturan, atau konfigurasi server berbeda. Gunakan hanya
untuk demo/test, bukan sebagai contoh keputusan klinis atau bukti fraud.
