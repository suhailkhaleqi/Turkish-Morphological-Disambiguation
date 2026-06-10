# Türkçe Morfolojik Çözümleme (Morphological Disambiguation) Projesi

## Genel Açıklama

Bu proje, **Turkish BOUN UD treebank** veri setini kullanarak Türkçe cümlelerdeki her kelimenin morfolojik etiketini (POS tag + morfolojik özellikler) doğru şekilde belirlemeyi amaçlamaktadır.

**Seçilen Yöntem:** Morfolojik Çözümleme (Morphological Disambiguation)
- CoNLL-U formatındaki veriden eğitim/test ayrımı yapılır
- Her kelime için bağlam-bağımlı özellikler çıkarılır
- Makine öğrenmesi modeli ile UPOS (Part-of-Speech) + morfolojik özellikler tahmin edilir
- Sonuçlar F-measure, precision, recall, accuracy ve confusion matrix ile değerlendirilir

## Projede İsterler Detayları

> [!IMPORTANT]
> **İster 1:** Herhangi bir istatistiksel makine öğrenmesi kullanılabilir → **Multinomial Logistic Regression** seçildi (hızlı, yorumlanabilir, CoNLL çıktısına uygun)
> 
> **İster 2:** Tüm işaretlemeler CoNLL formatında yapılmalıdır (B/I etiketleme de dahil)
> 
> **İster 3:** Çalışır kod ve dosyası eğitim/test işlemini yerine getirecek şekilde hazırlanmalıdır
> 
> **İster 4:** F-measure, precision, recall, accuracy + confusion matrix her sınıf için hesaplanmalı ve grafik olarak gösterilmelidir

## Önerilen Değişiklikler

### Proje Dosya Yapısı

```
donem projesi/
├── tr_boun-ud-train.conllu    (mevcut - eğitim verisi)
├── tr_boun-ud-test.conllu     (mevcut - test verisi)
├── morphological_disambiguator.py    [YENİ] Ana Python kodu
├── requirements.txt                  [YENİ] Bağımlılıklar
├── run_project.bat                   [YENİ] Windows çalıştırma scripti
├── output/                           [YENİ] Çıktı klasörü
│   ├── predictions_test.conllu       Tahmin edilen CoNLL-U çıktısı
│   ├── classification_report.txt     Sınıflandırma raporu
│   ├── confusion_matrix.png          Karışıklık matrisi görseli
│   ├── metrics_per_class.png         Sınıf bazlı F1/Precision/Recall grafiği
│   └── overall_metrics.png           Genel başarı özet grafiği
```

---

### Modül 1 - Veri Okuyucu (CoNLL-U Parser)

#### [YENİ] morphological_disambiguator.py

**Veri okuma:**
- Multi-word token satırlarını (ID'de tire olan) atla
- Her token için: `ID, FORM, LEMMA, UPOS, XPOS, FEATS` al
- Cümle bağlamını koru (önceki/sonraki kelimeler)

**Özellik Mühendisliği (Feature Engineering):**
Her token için şu özellikler çıkarılır:
- `form` (kelimenin kendisi - küçük harf)
- `lemma` (kök kelime)
- `suffix_1/2/3/4` (son 1-4 karakter - Türkçe sondan eklemeli yapı için kritik)
- `prefix_1/2/3` (ilk 1-3 karakter)
- `is_upper` (büyük harfle başlıyor mu - özel isim?)
- `is_digit` (rakam mı?)
- `has_apostrophe` (kesme işareti var mı? - Türkçe'de özel isimler)
- `word_length` (kelime uzunluğu)
- `prev_upos`, `prev2_upos` (önceki 2 kelimenin UPOS'u - bağlam)
- `next_upos` (sonraki kelimenin UPOS'u - oracle özellik, sadece eğitimde)
- `prev_form`, `next_form` (komşu kelime formları)
- `contains_vowel_harmony_pattern` (Türkçe sesli uyumu deseni)

**Model:**
- `sklearn.linear_model.LogisticRegression` (multinomial)
- `DictVectorizer` ile sparse feature matrix
- Sınıflar: Her UPOS etiketi (NOUN, VERB, ADJ, ADV, PRON, DET, ADP, CCONJ, SCONJ, AUX, PART, NUM, INTJ, PROPN, PUNCT, X, ...)

**Morfolojik Özellik Tahmini:**
- UPOS tahminine ek olarak, FEATS (morfolojik özellikler) de tahmin edilir
- FEATS için ayrı bir model ya da kural tabanlı yaklaşım kullanılır

**CoNLL-U Çıktısı (İster 2):**
- B-TAG / I-TAG formatında ek sütun veya tam CoNLL-U formatında tahmin çıktısı üretilir
- Tahmin edilen UPOS ve FEATS CoNLL-U dosyasına yazılır

**Değerlendirme (İster 4):**
- `sklearn.metrics.classification_report` → Precision, Recall, F1 per class
- `sklearn.metrics.confusion_matrix` → Seaborn heatmap ile görsel
- `sklearn.metrics.accuracy_score` → Genel doğruluk
- Macro/Micro/Weighted average hesaplaması

---

## Doğrulama Planı

### Otomatik Testler
```powershell
cd "d:\3_sinif_bahar_donemi\dogal dil islemeye giris\donem projesi"
pip install -r requirements.txt
python morphological_disambiguator.py
```

### Manuel Doğrulama
- `output/predictions_test.conllu` dosyasının CoNLL-U formatında olduğunu kontrol et
- Confusion matrix görselinin oluşturulduğunu doğrula
- Classification report'taki F1 skorlarını incele

## Teknik Notlar

> [!NOTE]
> **Zemberek hakkında:** Proje PDF'inde Zemberek yazılımından bahsediliyor. Zemberek, Java tabanlı bir Türkçe NLP kütüphanesidir. Bu projede **Zemberek'in Python wrapper'ı (zeyrek)** kullanılabilir, ancak önce CoNLL-U verisinden doğrudan eğitim yapılacak; Zemberek çok sayıda olası çözümleme üretmek için kullanılabilir (oracle yaklaşımı).
>
> CoNLL-U datasında zaten doğru etiketler mevcut olduğundan (gold standard), Zemberek olmadan da başarılı bir model eğitilebilir. Zemberek entegrasyonu opsiyonel kılınmıştır.

> [!WARNING]
> `tr_boun-ud-train.conllu` çok büyük (~127K satır). Scikit-learn modelleri bu boyutu handle eder ama eğitim birkaç dakika sürebilir.
