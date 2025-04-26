# Laporan Proyek Machine Learning - Bagus Angkasawan Sumantri Putra

## Domain Proyek

Obesitas adalah masalah kesehatan yang makin mengkhawatirkan secara global. Menurut WHO, pada tahun 2016 lebih dari 1,9 miliar orang dewasa mengalami kelebihan berat badan, dan lebih dari 650 juta di antaranya mengalami obesitas. Kondisi ini meningkatkan risiko penyakit kronis seperti diabetes, jantung, dan kanker.

Pendeteksian dini melalui machine learning dapat membantu klasifikasi tingkat obesitas seseorang berdasarkan atribut personal seperti usia, jenis kelamin, tinggi badan, berat badan, BMI, dan tingkat aktivitas fisik, sehingga tindakan preventif dapat dilakukan lebih cepat.

**Referensi:**
- [World Health Organization - Obesity and Overweight](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight)

## Business Understanding

### Problem Statements
- Bagaimana mengklasifikasikan tingkat obesitas seseorang berdasarkan data personal dan aktivitas fisiknya?
- Model machine learning apa yang paling akurat dalam melakukan klasifikasi ini?

### Goals
- Membuat model klasifikasi obesitas yang efektif dan akurat.
- Membandingkan performa berbagai model machine learning untuk memilih model terbaik.

### Solution Statements
- Menggunakan algoritma Logistic Regression, Random Forest, dan SVM untuk membangun model klasifikasi.
- Melakukan preprocessing dan tuning untuk meningkatkan performa model.
- Menggunakan classification report, confusion matrix, dan akurasi sebagai metrik evaluasi.

## Data Understanding

Dataset yang digunakan berasal dari [Kaggle - Obesity Level Prediction Dataset](https://www.kaggle.com/datasets/mrsimple07/obesity-prediction).

### Variabel dalam dataset:
- `Gender` : Jenis kelamin (Male/Female)
- `Age` : Usia
- `Height` : Tinggi badan (meter)
- `Weight` : Berat badan (kg)
- `BMI` : Body Mass Index
- `PhysicalActivityLevel` : Tingkat aktivitas fisik
- `ObesityCategory` : Kategori obesitas (target)

### Exploratory Data Analysis (EDA)
- Tidak ada missing value ditemukan.
- Distribusi kategori `ObesityCategory` cenderung tidak seimbang.
- Korelasi kuat antara `Weight` dan `BMI` ditemukan.

## Data Preparation

Langkah yang dilakukan:
- **Encoding**: Label Encoding pada kolom kategorikal seperti `Gender` dan `ObesityCategory`.
- **Scaling**: Standardisasi fitur numerik menggunakan StandardScaler.
- **Splitting**: Membagi data menjadi training dan testing set (80%:20%).

Tahapan ini penting untuk memastikan model dapat bekerja optimal, khususnya model SVM yang sensitif terhadap skala data.

## Modeling

Model yang digunakan:
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Support Vector Machine (SVM)**

**Keterangan:**
- Logistic Regression dijadikan baseline model.
- Random Forest dipilih karena kekuatannya dalam menangani data kompleks dan mengurangi overfitting.
- SVM digunakan untuk menangani klasifikasi dengan margin maksimal.

Dalam hal ini, **Random Forest** dipilih sebagai model terbaik karena memberikan akurasi tertinggi (99.5%) bahkan sebelum tuning. Meskipun Logistic Regression juga memberikan performa yang sangat baik dengan akurasi 97%, Random Forest lebih stabil dalam menangani kompleksitas data dan dapat mengurangi kemungkinan overfitting dengan banyaknya pohon yang digunakan. Oleh karena itu, Random Forest lebih diandalkan untuk mencapai performa yang optimal dalam prediksi obesitas.

## Evaluation

Metrik evaluasi:
- **Classification Report**: Precision, Recall, F1-score.
- **Confusion Matrix**: Visualisasi prediksi benar/salah.
- **Accuracy**: Skor akurasi total.

### Formula Metrik:
- **Accuracy**: Persentase prediksi yang benar dibandingkan dengan total data.  
  Formula:  
  `Accuracy = (True Positives + True Negatives) / Total Samples`

- **Precision**: Mengukur akurasi dari prediksi yang positif.  
  Formula:  
  `Precision = True Positives / (True Positives + False Positives)`

- **Recall (Sensitivity)**: Mengukur seberapa baik model menangkap kelas positif yang sesungguhnya.  
  Formula:  
  `Recall = True Positives / (True Positives + False Negatives)`

- **F1-Score**: Rata-rata harmonis dari Precision dan Recall.  
  Formula:  
  `F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`

  
### Hasil Evaluasi Sebelum Tuning:
| Model               | Akurasi |
|---------------------|---------|
| Logistic Regression | 0.97    |
| Random Forest       | 0.995   |
| SVM                 | 0.93    |

**Keterangan:**
- **Random Forest** memberikan akurasi tertinggi (99.5%) bahkan sebelum tuning, dengan F1-score yang hampir sempurna di semua kelas.
- **Logistic Regression** sedikit lebih rendah (97%), namun masih menunjukkan performa yang sangat baik.

**Confusion Matrix Random Forest (sebelum tuning):**
```
[[74  0  0  0]
 [ 0 37  1  0]
 [ 0  0 59  0]
 [ 0  0  0 29]]
```

### Tuning dan Evaluasi Random Forest

Dilakukan tuning pada:
- `n_estimators = 200`
- `max_depth = 10`
- `random_state = 42`

Model **Tuned Random Forest** ini dilatih ulang dan dievaluasi.

### Hasil Setelah Tuning:
- **Accuracy**: 0.995 (masih sangat tinggi)
- **F1-score** di semua kategori tetap sangat baik.
- **Confusion Matrix** tetap menunjukkan model mendeteksi hampir semua kelas dengan sempurna.

**Confusion Matrix Tuned Random Forest:**
```
[[74  0  0  0]
 [ 0 37  1  0]
 [ 0  0 59  0]
 [ 0  0  0 29]]
```

### Kesimpulan Evaluasi:
- Random Forest (baik sebelum maupun sesudah tuning) adalah model terbaik untuk kasus ini.
- Logistic Regression dan SVM juga memiliki performa bagus, tetapi sedikit di bawah Random Forest.
- Dengan tuning sederhana, performa Random Forest tetap stabil di tingkat akurasi hampir sempurna.
