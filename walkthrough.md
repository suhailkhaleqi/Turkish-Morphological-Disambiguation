# Türkçe Morfolojik Çözümleme - Proje Tamamlandı

## Sonuçlar

| Metrik | Değer |
|---|---|
| **Accuracy (Doğruluk)** | **92.24%** |
| Macro F1-Score | 84.60% |
| Weighted F1-Score | 92.18% |
| Macro Precision | 87.59% |
| Macro Recall | 83.14% |
| Eğitim Süresi | 17.3 saniye |
| Toplam Süre | 44.2 saniye |

## Veri Seti

| | Cümle | Token |
|---|---|---|
| **Eğitim** | 7,803 | 100,713 |
| **Test** | 979 | 12,210 |
| **Özellik boyutu** | — | 177,458 |

## Projede İsterler - Kontrol Listesi

- [x] **İster 1:** İstatistiksel makine öğrenmesi → **Multinomial Logistic Regression** (SAGA solver)
- [x] **İster 2:** Tüm işaretlemeler CoNLL formatında → `output/predictions_test.conllu` (BIO etiketleme dahil)
- [x] **İster 3:** Çalışır kod eğitim ve test işlemini yerine getiriyor → `morphological_disambiguator.py`
- [x] **İster 4:** F-measure, precision, recall, accuracy + confusion matrix grafikler

## Sınıf Bazlı Sonuçlar

| UPOS | Precision | Recall | F1 | Destek |
|---|---|---|---|---|
| PUNCT | 1.00 | 1.00 | **1.00** | 2,028 |
| DET | 0.96 | 0.99 | **0.97** | 546 |
| CCONJ | 0.95 | 0.93 | **0.94** | 337 |
| VERB | 0.95 | 0.92 | **0.94** | 2,199 |
| NUM | 0.92 | 0.92 | **0.92** | 276 |
| NOUN | 0.90 | 0.93 | **0.92** | 3,955 |
| PRON | 0.93 | 0.91 | **0.92** | 317 |
| ADP | 0.96 | 0.84 | **0.90** | 264 |
| AUX | 0.87 | 0.90 | **0.88** | 240 |
| PROPN | 0.86 | 0.87 | **0.87** | 677 |
| ADV | 0.85 | 0.80 | **0.82** | 480 |
| ADJ | 0.84 | 0.78 | **0.81** | 681 |
| PART | 0.83 | 0.97 | **0.89** | 162 |
| SCONJ | 0.83 | 0.38 | **0.53** | 26 |
| INTJ | 0.50 | 0.32 | **0.39** | 22 |

> [!NOTE]
> SCONJ ve INTJ sınıflarının düşük skoru, test setinde çok az örnek bulunmasından kaynaklanmaktadır (26 ve 22 örnek).

## Oluşturulan Dosyalar

| Dosya | Açıklama |
|---|---|
| [morphological_disambiguator.py](file:///d:/3_sinif_bahar_donemi/dogal%20dil%20islemeye%20giris/donem%20projesi/morphological_disambiguator.py) | Ana Python kodu |
| [requirements.txt](file:///d:/3_sinif_bahar_donemi/dogal%20dil%20islemeye%20giris/donem%20projesi/requirements.txt) | Bağımlılıklar |
| [run_project.bat](file:///d:/3_sinif_bahar_donemi/dogal%20dil%20islemeye%20giris/donem%20projesi/run_project.bat) | Windows çalıştırma scripti |
| [output/predictions_test.conllu](file:///d:/3_sinif_bahar_donemi/dogal%20dil%20islemeye%20giris/donem%20projesi/output/predictions_test.conllu) | CoNLL-U formatında tahminler (BIO etiketli) |
| [output/classification_report.txt](file:///d:/3_sinif_bahar_donemi/dogal%20dil%20islemeye%20giris/donem%20projesi/output/classification_report.txt) | Sınıflandırma raporu |
| [output/confusion_matrix.png](file:///d:/3_sinif_bahar_donemi/dogal%20dil%20islemeye%20giris/donem%20projesi/output/confusion_matrix.png) | Karışıklık matrisi |
| [output/metrics_per_class.png](file:///d:/3_sinif_bahar_donemi/dogal%20dil%20islemeye%20giris/donem%20projesi/output/metrics_per_class.png) | Sınıf bazlı metrikler |
| [output/overall_metrics.png](file:///d:/3_sinif_bahar_donemi/dogal%20dil%20islemeye%20giris/donem%20projesi/output/overall_metrics.png) | Genel başarı özeti |

## Teknik Yaklaşım

### Özellik Mühendisliği (Feature Engineering)
- **Sonekler (suffix_1..5):** Türkçe sondan eklemeli yapı için kritik
- **Önekler (prefix_1..3):** Büyük harf başlangıcı (özel isimler)
- **Bağlam özellikleri:** Önceki/sonraki 2 token (form + UPOS)
- **Morfolojik ipuçları:** Türkçe'ye özgü ek desenleri (-yor, -mış, -dı, -lık)
- **Karakter özellikleri:** Büyük harf, rakam, kesme işareti, uzunluk kategorisi

### Model
- **Multinomial Logistic Regression** (scikit-learn, SAGA solver)
- **177,458 özellik boyutu** ile sparse feature matrix
- **16 UPOS sınıfı** (ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, VERB, X)

## Projeyi Çalıştırma

```powershell
cd "d:\3_sinif_bahar_donemi\dogal dil islemeye giris\donem projesi"
python -X utf8 morphological_disambiguator.py
```

Ya da çift tıklayarak: `run_project.bat`
