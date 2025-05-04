# Laporan Proyek Machine Learning - Muhammad Aditya Bayhaqie

## Domain Proyek

Dalam dunia pendidikan, pemantauan kinerja siswa menjadi faktor penting dalam meningkatkan kualitas pembelajaran. Dengan menggunakan teknik Machine Learning, kita dapat memprediksi performa siswa berdasarkan berbagai faktor seperti demografi, akademik, dan finansial. Model prediksi ini dapat membantu lembaga pendidikan dalam memberikan intervensi yang lebih tepat kepada siswa yang berisiko mengalami kesulitan akademik.

Beberapa penelitian menunjukkan bahwa faktor-faktor seperti latar belakang keluarga, partisipasi dalam beasiswa, dan pola belajar siswa dapat berpengaruh signifikan terhadap performa akademik mereka

- ([Predicting Student Performance](https://github.com/Damiieibikun/Student-s-Dropout-Prediction-using-Supervised-Machine-Learning-Classifiers/tree/main)).
- ([Predicting student dropouts with machine learning: An empirical study in
  Finnish higher education](https://pdf.sciencedirectassets.com/271744/1-s2.0-S0160791X23X00050/1-s2.0-S0160791X24000228/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEGcaCXVzLWVhc3QtMSJHMEUCIC%2FFu5SKlZPu8t1lbHJepUR%2FMI5qZe09huubOlPZ5v5BAiEA2Ods5dqozYvHTnz4Yb4KNcfZNFNCZC%2F%2F%2FDK9C%2BHprzEquwUI0P%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDEC%2FQkJ7nopowfA0ryqPBRwDd41E14%2BLa0MJkFxnuagv7%2BBg%2Bi6p%2FKA1M5hH2fEAGYtdeMtRr%2F%2BEV8a9vIiUvVmW50zMfqsydK3QefTSDN8IvBrrulPIfWU0EGnD4yx2mm5tFI8yyQ8xROb9UHvL7tz%2FdGFRD8pwyO8mQ1Ostc9bBcXk1NNFrAwv%2FYBI4l%2FKQNNgh65vgafX4n5EBEmx2qdUJQ1WKg%2FoZOU%2FjSHcqOqD3DbOvkXEa0OKIt0dc2IXE0ZJpxdN%2BE6KuC0MeG5TD%2BBmyW%2B6McT5htWOQ6KAOnj14GmjpUx9lZWwbRl94Ll0yo%2FjYZ3vMxhkgu5PKASk0z3G0MABYcBsqqNFGkuyQ0zTd858%2Bphq9tHKuFjRuf%2B4Y8t%2FVE4w4zZaO9T%2BsP%2BC3SBnYPDE%2B%2F3amJ2mbpUPD23RD6OVQ6%2FWT7yb43FU%2Bm4z0B%2Fy1vmSvwoKu7QI10J2M4eNZHUGlcw0GMGM5eiSaIzgRwb%2FzISi5s%2BmjZzqCy8lRjVp4WCCLJzsay0DuDIGIJJntlhlyVsANveHJdyUSTG6d1myF9TSyxULJgv%2ByX5KHbFEaN53bpJukDCT7ZurCSKYKhm6PS1AjOvl54eK7WtMsIhO1AiQ4uEd6ObIgsfz7fDiA6SlilDyLRveW%2FtcTA028whsb5fvI72Xrb%2BpLV%2FnQFh1tWQGWHtiXTgTGgWjYlDrnYQjwerD6X2Pfr%2FjFPiR%2B4FiT26gD4M47zXvAOjWe4%2BJ9QyDZAyT6b8gwuKf49LIIIZYQ1d9A%2FsNL%2FFJHYz%2FyAWWkxTYv%2F6oSlHhG9TxegTImlrkVc0Tm3rrpBpJn5x%2FXNoyeJCIjtMVbFzK5Xad5fLKfAFpCs31KtRChdkHkPIdM%2BpFsk%2FSX86McE4wtL6zvwY6sQGwlhNTwqjiksptmuwkYM5jF31cSq4THP0uJ6RxLGiE19EZ6FhESPiXu56t%2BShhvl1Xs6FprQkVoAgIQnkDNQQKsi5lUxwCXAql5Ao%2Bipcrbf9bRKD91BASAJOQZ6D2WSdqpO%2FPTEADthgN0XWSDyLSGloLHezN4rTU7ioBR3FR7dnSpJ63lzN4vOg80Crmrx4nCcNRXtTPrdBIFCSYPNLWkHXC7QuJy6lWMoF%2Bt%2ByU6Ls%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250402T074701Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYS2KTJ6ZX%2F20250402%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=f9da16242529aac19c7449c6251a63b4b78bc517d4b15f749fc2bfc58dc033da&hash=ca679eff5931146f40baaab14410f5085fa2909bb113188567de9f88629945a6&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0160791X24000228&tid=spdf-19f41f66-464d-4983-80d0-e8fe9038192a&sid=72bf58d51ba29745746ae3a5050110ba8488gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=1302575159050056570707&rr=929eb73acd924b6d&cc=id)).

## Business Understanding

### Problem Statements

1. Bagaimana cara memprediksi performa akademik siswa berdasarkan faktor-faktor yang tersedia?
2. Faktor apa saja yang memiliki pengaruh signifikan terhadap keberhasilan akademik siswa?
3. Model Machine Learning mana yang paling efektif untuk memprediksi performa siswa?

### Goals

1. Mengembangkan model prediktif untuk memproyeksikan performa siswa berdasarkan data historis.
2. Mengidentifikasi faktor-faktor yang memiliki dampak paling besar terhadap hasil akademik siswa.
3. Mengevaluasi dan memilih model terbaik berdasarkan metrik evaluasi yang sesuai.

### Solution Statements

- Menggunakan berbagai algoritma Machine Learning seperti Gradient Boosting, Logistic Regression, dan Support Vector Machine (SVM) untuk memprediksi performa siswa.
- Melakukan eksplorasi data dan visualisasi untuk memahami hubungan antar fitur.
- Melakukan hyperparameter tuning untuk meningkatkan performa model.
- Membandingkan model berdasarkan metrik evaluasi seperti akurasi, precision, recall, dan F1-score.

## Data Understanding

### URL/Tautan Sumber Data

Dataset yang digunakan dalam proyek ini diambil dari [Predict Students Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) Dari UCI Dataset. Dataset ini mencakup berbagai faktor yang berhubungan dengan performa akademik siswa.

### Jumlah Baris dan Kolom

Dataset ini terdiri dari:

- **4.424 baris** (data mahasiswa)

- **37 kolom** (fitur)

### Kondisi Data

| Fitur                                              | Deskripsi                                                                         |
| -------------------------------------------------- | --------------------------------------------------------------------------------- |
| `Marital Status`                                 | Status pernikahan mahasiswa (1 = belum menikah, 2 = menikah, dst.)                |
| `Application Mode`                               | Metode/jenis pendaftaran yang digunakan oleh mahasiswa                            |
| `Application Order`                              | Urutan pilihan program studi saat mendaftar                                       |
| `Course`                                         | ID program studi yang dipilih                                                     |
| `Daytime/evening attendance`                     | Jenis kelas yang diikuti (1 = kelas pagi, 0 = kelas malam)                        |
| `Previous qualification`                         | Kualifikasi akademik sebelumnya                                                   |
| `Previous qualification (grade)`                 | Nilai dari pendidikan sebelumnya                                                  |
| `Nacionality`                                    | Kode negara kewarganegaraan                                                       |
| `Mother's qualification`                         | Tingkat pendidikan ibu                                                            |
| `Father's qualification`                         | Tingkat pendidikan ayah                                                           |
| `Mother's occupation`                            | Jenis pekerjaan ibu                                                               |
| `Father's occupation`                            | Jenis pekerjaan ayah                                                              |
| `Admission grade`                                | Nilai saat masuk universitas                                                      |
| `Displaced`                                      | Apakah mahasiswa berasal dari wilayah terdampak atau pindahan (1 = ya, 0 = tidak) |
| `Educational special needs`                      | Apakah memiliki kebutuhan khusus (1 = ya, 0 = tidak)                              |
| `Debtor`                                         | Apakah mahasiswa memiliki utang (1 = ya, 0 = tidak)                               |
| `Tuition fees up to date`                        | Status pembayaran uang kuliah (1 = ya, 0 = tidak)                                 |
| `Gender`                                         | Jenis kelamin (1 = laki-laki, 2 = perempuan)                                      |
| `Scholarship holder`                             | Apakah mahasiswa penerima beasiswa (1 = ya, 0 = tidak)                            |
| `Age at enrollment`                              | Usia mendaftar kuliah                                                             |
| `International`                                  | Apakah mahasiswa internasional (1 = ya, 0 = tidak)                                |
| `Curricular units 1st sem (credited)`            | Jumlah SKS yang diakui di Semester 1                                              |
| `Curricular units 1st sem (enrolled)`            | Jumlah mata kuliah yang diambil di semester 1                                     |
| `Curricular units 1st sem (evaluations)`         | Jumlah evaluasi yang diikuti di semester 1                                        |
| `Curricular units 1st sem (approved)`            | Jumlah mata kuliah yang lulus di semester 1                                       |
| `Curricular units 1st sem (grade)`               | Rata-rata nilai di semester 1                                                     |
| `Curricular units 1st sem (without evaluations)` | Jumlah mata kuliah yang tidak diikuti evaluasinya                                 |
| `Curricular units 2nd sem (credited)`            | Jumlah SKS yang diakui di Semester 2                                              |
| `Curricular units 2nd sem (evaluations)`         | Jumlah mata kuliah yang diambil di semester 2                                     |
| `Curricular units 2nd sem (approved)`            | Jumlah evaluasi yang diikuti di semester 2                                        |
| `Curricular units 2nd sem (grade)`               | Jumlah mata kuliah yang lulus di semester 2                                       |
| `Curricular units 2nd sem (without evaluations)` | Rata-rata nilai di semester 2                                                     |
| `Unemployment rate`                              | Tingkat pengangguran di tahun tersebut                                            |
| `Inflation rate`                                 | Tingkat inflasi ekonomi pada tahun tersebut                                       |
| `GDP`                                            | Pertumbuhan Produk Domestik Bruto saat itu                                        |
| `Target`                                         | Status akhir mahasiswa (`Dropout`, `Graduate`, `Enrolled`)                  |

### Fitur dalam Dataset:

- Exploratory Data Analysis (EDA)
- Visualisasi distribusi fitur menggunakan histogram dan boxplot.
- Korelasi antar fitur menggunakan heatmap untuk mengidentifikasi fitur yang memiliki hubungan kuat.

## Data Preparation

- **Handling Missing Values**: Mengisi nilai yang hilang dengan mean/median untuk variabel numerik dan modus untuk variabel kategorikal. (Tidak ada Missing Values, sehingga tahapan ini akan dilewati)
- **Handling Outliers**: Mengidentifikasi dan menangani outlier menggunakan metode seperti IQR (Interquartile Range) atau Z-score untuk meningkatkan kualitas data.
- **Feature Encoding**: Mengonversi variabel kategorikal ke dalam bentuk numerik menggunakan One-Hot Encoding agar dapat digunakan dalam model machine learning. (Data telah di Encode, sehingga tahapan ini akan dilewati)
- **Dimensionality Reduction (PCA)**: Menggunakan Principal Component Analysis untuk mereduksi jumlah fitur dan menghindari overfitting.
- **Train-Test Split**: Membagi dataset menjadi data latih dan data uji (misalnya 80:20) untuk mengevaluasi performa model secara objektif.
- **Feature Scaling**: Normalisasi menggunakan StandardScaler untuk memastikan distribusi fitur numerik menjadi seimbang dan setara.

Tahapan ini dilakukan secara runtut untuk memastikan data yang digunakan dalam pelatihan model berada dalam kondisi optimal dan representatif terhadap permasalahan yang ingin diselesaikan.

## Modeling

### Model yang digunakan:

1. **Gradient Boosting**

   *Cara Kerja:*

   Gradient Boosting bekerja dengan membangun model secara bertahap, di mana setiap model baru mencoba memperbaiki kesalahan dari model sebelumnya. Model ini menggunakan teknik boosting yang menggabungkan beberapa pohon keputusan lemah menjadi satu model prediktif yang kuat.
   *Parameter:*

   * `n_estimators`: jumlah pohon yang digunakan.
   * `learning_rate`: seberapa besar kontribusi setiap pohon terhadap model akhir.
   * `max_depth`: kedalaman maksimal setiap pohon.
   * `subsample`: proporsi sampel yang digunakan untuk membangun setiap pohon.

   *Kelebihan:*

   - Mampu menangkap pola yang kompleks dan bekerja baik pada data tabular

   *Kekurangan:*

   * Waktu pelatihan relatif lebih lama.
2. **Logistic Regression**

   *Cara Kerja:*

   Logistic Regression adalah model linear yang digunakan untuk klasifikasi biner. Model ini menghitung probabilitas dari kelas target menggunakan fungsi sigmoid terhadap kombinasi linear dari fitur input.
   *Parameter:*

   * `penalty`: jenis regularisasi yang digunakan (L1, L2, atau ElasticNet).
   * `C`: parameter regularisasi, semakin kecil nilai C, semakin kuat regularisasi.
   * `solver`: algoritma optimisasi yang digunakan untuk menemukan parameter terbaik.

   *Kelebihan:*

   - Sederhana dan mudah diinterpretasikan.

   *Kekurangan:*

   - Tidak bekerja dengan baik pada data non-linear.
3. **Support Vector Machine (SVM)**

   *Cara Kerja:*

   SVM mencari hyperplane terbaik yang memisahkan kelas-kelas dalam ruang fitur. Untuk data non-linear, SVM menggunakan fungsi kernel untuk memetakan data ke dimensi yang lebih tinggi agar dapat dipisahkan secara linear.
   *Parameter:*

   * `kernel`: tipe fungsi kernel yang digunakan (linear, poly, rbf, sigmoid).
   * `C`: parameter regularisasi yang mengontrol trade-off antara margin maksimum dan kesalahan klasifikasi.
   * `gamma`: menentukan seberapa jauh pengaruh satu data terhadap yang lain (hanya untuk kernel non-linear).

   *Kelebihan:*

   - Bekerja dengan baik pada data non-linear menggunakan kernel.

   *Kekurangan:*

   * Tidak skala dengan baik pada dataset besar.
4. **Random Forest**

   *Cara Kerja:*

   Random Forest adalah ensemble dari pohon keputusan yang dilatih menggunakan teknik bagging. Setiap pohon dibangun dari subset data acak, dan hasil akhir adalah rata-rata (untuk regresi) atau voting mayoritas (untuk klasifikasi) dari semua pohon.
   *Parameter:*

   * `n_estimators`: jumlah pohon dalam hutan.
   * `max_depth`: kedalaman maksimal setiap pohon.
   * `max_features`: jumlah fitur yang dipertimbangkan saat mencari split terbaik.
   * `bootstrap`: apakah akan menggunakan bootstrap sampling.

   *Kelebihan:*

   - Dapat menangani overfitting dengan baik dan efektif pada dataset dengan banyak fitur.

   *Kekurangan:*

   * Model bisa menjadi kompleks dan lambat saat jumlah pohon terlalu banyak.

## Evaluation

Metrik yang digunakan dalam proyek ini adalah:

1. **Akurasi** = (TP + TN) / (TP + TN + FP + FN)
2. **Precision** = TP / (TP + FP)
3. **Recall** = TP / (TP + FN)
4. **F1-score** = 2 * (Precision * Recall) / (Precision + Recall)

### Hasil Evaluasi Model:

| Model               | Akurasi | Precision | Recall | F1-score |
| ------------------- | ------- | --------- | ------ | -------- |
| Gradient Boosting   | 77%     | 76%       | 77%    | 77%      |
| Logistic Regression | 75%     | 72%       | 75%    | 72%      |
| SVM                 | 74%     | 72%       | 74%    | 71%      |
| Random Forest       | 76%     | 73%       | 76%    | 74%      |

Model terbaik yang dipilih adalah **Gradient Boosting**, karena memberikan akurasi dan F1-score tertinggi dibandingkan model lainnya, namun model ini masih perlu dikembangkan.

## Kesimpulan

- Model Gradient Boosting memberikan hasil terbaik dalam memprediksi performa siswa.
- Faktor-faktor seperti **Previous Grades** dan **Study Hours** memiliki pengaruh paling signifikan terhadap performa siswa.
- Model ini dapat digunakan oleh institusi pendidikan untuk memberikan intervensi dini kepada siswa yang berisiko mengalami kesulitan akademik.

## Rekomendasi

- **Penggunaan Model dalam Sistem Nyata**: Model dapat diintegrasikan dalam sistem manajemen akademik untuk memberikan peringatan dini kepada siswa yang berisiko.
- **Penelitian Lanjutan**: Menganalisis bagaimana faktor sosial dan ekonomi memengaruhi performa akademik.
- **Perkembangan Model**: Banyak model yang dapat digunakan untuk mengembangkan model ini, sehingga nantinya peneliti dapat mengembangkan model ini dengan model lainnya.

---
