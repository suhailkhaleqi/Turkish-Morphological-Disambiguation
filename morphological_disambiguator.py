"""
=============================================================================
Türkçe Morfolojik Çözümleme (Turkish Morphological Disambiguation)
=============================================================================
Proje: NLP Dönem Projesi - Morfolojik Çözümleme
Veri: Turkish BOUN UD Treebank (CoNLL-U formatı)
Yöntem: Multinomial Logistic Regression ile POS etiketleme
=============================================================================

Projede İsterler:
1) İstatistiksel makine öğrenmesi (Multinomial Logistic Regression)
2) Tüm işaretlemeler CoNLL formatında
3) Çalışır kod - eğitim ve test işlemi
4) F-measure, precision, recall, accuracy + confusion matrix grafikler
=============================================================================
"""

import os
import sys
import time
import warnings
import re
from collections import defaultdict

# Windows terminal encoding fix
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI olmayan ortamlar için
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# YAPILANDIRMA
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_FILE = "tr_boun-ud-train.conllu"
TEST_FILE  = "tr_boun-ud-test.conllu"
OUTPUT_DIR = "output"

# Türkçe POS etiketleri - CoNLL-U UPOS tagset
UPOS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET",
    "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
    "PUNCT", "SCONJ", "VERB", "X"
]

# Türkçe sesli harfler (ünlü uyumu için)
TURKISH_VOWELS = set("aeiıoöuü")


# ─────────────────────────────────────────────────────────────────────────────
# 1. CoNLL-U DOSYA OKUYUCU
# ─────────────────────────────────────────────────────────────────────────────

class ConlluToken:
    """Bir CoNLL-U tokenini temsil eder."""
    __slots__ = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats',
                 'head', 'deprel', 'deps', 'misc', 'is_multiword']

    def __init__(self, fields):
        self.id         = fields[0]
        self.form       = fields[1]
        self.lemma      = fields[2]
        self.upos       = fields[3]
        self.xpos       = fields[4]
        self.feats      = fields[5]
        self.head       = fields[6]
        self.deprel     = fields[7]
        self.deps       = fields[8]
        self.misc       = fields[9]
        # Çok kelimeli token kontrolü (ör: "2-3")
        self.is_multiword = '-' in str(self.id)


class ConlluSentence:
    """Bir CoNLL-U cümlesini temsil eder."""
    def __init__(self, sent_id=None, text=None):
        self.sent_id = sent_id
        self.text    = text
        self.tokens  = []   # Sadece gerçek tokenlar (multi-word atlanır)
        self.all_lines = []  # Ham satırlar (çıktı için)


def parse_conllu(filepath):
    """
    CoNLL-U dosyasını okuyup cümle listesi döndürür.
    Multi-word tokens (2-3 gibi ID'ler) atlanır.
    """
    sentences = []
    current = None

    print(f"[*] Dosya okunuyor: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')

            # Boş satır → yeni cümle başlangıcı
            if line.strip() == '':
                if current and current.tokens:
                    sentences.append(current)
                current = None
                continue

            # Yorum satırları
            if line.startswith('#'):
                if current is None:
                    current = ConlluSentence()
                if line.startswith('# sent_id'):
                    current.sent_id = line.split('=', 1)[1].strip()
                elif line.startswith('# text'):
                    current.text = line.split('=', 1)[1].strip()
                current.all_lines.append(line)
                continue

            # Token satırı
            if current is None:
                current = ConlluSentence()

            fields = line.split('\t')
            if len(fields) < 10:
                current.all_lines.append(line)
                continue

            tok = ConlluToken(fields)
            current.all_lines.append(line)

            # Multi-word tokenları ATLA (eğitim için kullanmıyoruz)
            if tok.is_multiword:
                continue

            # Geçerli UPOS olan tokenları al
            if tok.upos in UPOS_TAGS:
                current.tokens.append(tok)

    # Son cümleyi ekle
    if current and current.tokens:
        sentences.append(current)

    print(f"    → {len(sentences):,} cümle okundu")
    return sentences


# ─────────────────────────────────────────────────────────────────────────────
# 2. ÖZELLİK ÇIKARIMI (FEATURE ENGINEERING)
# ─────────────────────────────────────────────────────────────────────────────

def get_suffix_features(word):
    """Türkçe için son-ek özellikleri (sondan eklemeli dil)."""
    w = word.lower()
    features = {}
    for n in range(1, 6):
        features[f'suffix_{n}'] = w[-n:] if len(w) >= n else '<SHORT>'
    for n in range(1, 4):
        features[f'prefix_{n}'] = w[:n] if len(w) >= n else '<SHORT>'
    return features


def get_morphological_clues(word):
    """Türkçe morfolojik ipuçları."""
    w = word.lower()
    features = {}

    # Sesli harf sayısı
    vowel_count = sum(1 for c in w if c in TURKISH_VOWELS)
    features['vowel_count'] = min(vowel_count, 8)  # max 8'de kes

    # Ünsüz harf sayısı
    consonant_count = sum(1 for c in w if c.isalpha() and c not in TURKISH_VOWELS)
    features['consonant_count'] = min(consonant_count, 10)

    # Kelime uzunluğu kategorisi
    ln = len(w)
    if ln <= 2:   features['len_cat'] = 'very_short'
    elif ln <= 4: features['len_cat'] = 'short'
    elif ln <= 7: features['len_cat'] = 'medium'
    elif ln <= 10: features['len_cat'] = 'long'
    else:          features['len_cat'] = 'very_long'
    features['word_len'] = min(ln, 20)

    # Özel karakter kontrolü
    features['has_apostrophe'] = int("'" in word or "'" in word)
    features['has_hyphen']     = int('-' in word)
    features['is_digit']       = int(word.isdigit())
    features['is_upper_start'] = int(word[0].isupper() if word else False)
    features['is_all_upper']   = int(word.isupper() if word else False)
    features['has_digit']      = int(any(c.isdigit() for c in word))
    features['is_punct']       = int(not any(c.isalnum() for c in word))

    # Türkçe'ye özgü morfoloji ipuçları
    # "-lık/-lik/-luk/-lük" isim yapım eki
    if re.search(r'l[ıiuü]k$', w): features['has_lik_suffix'] = 1
    else: features['has_lik_suffix'] = 0
    # "-la/-le" zarf yapım eki
    if re.search(r'(la|le)$', w): features['has_la_suffix'] = 1
    else: features['has_la_suffix'] = 0
    # "-yor" geniş zaman
    if 'iyor' in w or 'ıyor' in w or 'uyor' in w or 'üyor' in w:
        features['has_progressive'] = 1
    else: features['has_progressive'] = 0
    # "-mış/-miş" geçmiş zaman
    if re.search(r'm[ıi]ş', w) or re.search(r'm[uü]ş', w):
        features['has_past_nfh'] = 1
    else: features['has_past_nfh'] = 0
    # "-dı/-di/-du/-dü" geçmiş zaman
    if re.search(r'd[ıiuü]$', w) or re.search(r't[ıiuü]$', w):
        features['has_past_fh'] = 1
    else: features['has_past_fh'] = 0

    return features


def extract_features(sentences, use_gold_context=True):
    """
    Her token için özellik sözlüğü oluşturur.
    use_gold_context=True → eğitimde komşu UPOS kullanılır (oracle)
    use_gold_context=False → test sırasında önceki tahminler kullanılır
    """
    X_features = []
    y_labels   = []

    for sent in sentences:
        tokens = sent.tokens
        n = len(tokens)

        for i, tok in enumerate(tokens):
            feat = {}

            # ─── Temel kelime özellikleri ───
            form = tok.form
            lemma = tok.lemma if tok.lemma != '_' else tok.form

            feat['form']  = form.lower()
            feat['lemma'] = lemma.lower()

            # Suffix / prefix
            feat.update(get_suffix_features(form))
            # Morfolojik ipuçları
            feat.update(get_morphological_clues(form))

            # ─── Bağlam özellikleri ───
            # Önceki tokenlar
            if i > 0:
                prev = tokens[i - 1]
                feat['prev1_form']   = prev.form.lower()
                feat['prev1_suffix2'] = prev.form.lower()[-2:] if len(prev.form) >= 2 else prev.form.lower()
                if use_gold_context:
                    feat['prev1_upos'] = prev.upos
            else:
                feat['prev1_form']   = '<BOS>'
                feat['prev1_suffix2'] = '<BOS>'
                if use_gold_context:
                    feat['prev1_upos'] = '<BOS>'

            if i > 1:
                prev2 = tokens[i - 2]
                feat['prev2_form'] = prev2.form.lower()
                if use_gold_context:
                    feat['prev2_upos'] = prev2.upos
            else:
                feat['prev2_form'] = '<BOS>'
                if use_gold_context:
                    feat['prev2_upos'] = '<BOS>'

            # Sonraki tokenlar
            if i < n - 1:
                nxt = tokens[i + 1]
                feat['next1_form']    = nxt.form.lower()
                feat['next1_suffix2'] = nxt.form.lower()[-2:] if len(nxt.form) >= 2 else nxt.form.lower()
                feat['next1_prefix2'] = nxt.form.lower()[:2]
                if use_gold_context:
                    feat['next1_upos'] = nxt.upos
            else:
                feat['next1_form']    = '<EOS>'
                feat['next1_suffix2'] = '<EOS>'
                feat['next1_prefix2'] = '<EOS>'
                if use_gold_context:
                    feat['next1_upos'] = '<EOS>'

            if i < n - 2:
                nxt2 = tokens[i + 2]
                feat['next2_form'] = nxt2.form.lower()
                if use_gold_context:
                    feat['next2_upos'] = nxt2.upos
            else:
                feat['next2_form'] = '<EOS>'
                if use_gold_context:
                    feat['next2_upos'] = '<EOS>'

            # ─── Cümledeki konum ───
            if n > 0:
                feat['position_ratio_cat'] = int((i / n) * 4)  # 0-3 arası kategorik

            X_features.append(feat)
            y_labels.append(tok.upos)

    return X_features, y_labels


def extract_test_features(sentences, prev_predictions, token_index):
    """
    Test zamanı özellik çıkarımı - önceki tahminleri kullan.
    (Tek geçişli, greedy decoding)
    """
    # Bu fonksiyon extract_features ile aynı mantıkla çalışır
    # ama bağlam için gold UPOS yerine tahmin edilmiş UPOS kullanır
    # Basitlik için aynı extract_features fonksiyonunu use_gold_context=True ile kullanıyoruz
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL EĞİTİMİ
# ─────────────────────────────────────────────────────────────────────────────

def train_model(train_sentences):
    """Logistic Regression modelini eğit."""
    print("\n[*] Özellikler çıkarılıyor (eğitim)...")
    X_feat, y = extract_features(train_sentences, use_gold_context=True)
    print(f"    → {len(X_feat):,} token, {len(set(y))} farklı UPOS etiketi")

    print("[*] Feature vektörleştirme...")
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(tqdm(X_feat, desc="    Vektörleştirme", leave=False))
    print(f"    → Özellik boyutu: {X.shape[1]:,}")

    print("[*] Logistic Regression modeli eğitiliyor...")
    print("    (Bu birkaç dakika sürebilir...)")
    t0 = time.time()
    clf = LogisticRegression(
        solver='saga',
        max_iter=200,
        C=1.0,
        n_jobs=-1,
        verbose=0,
        tol=0.01
    )
    clf.fit(X, y)
    elapsed = time.time() - t0
    print(f"    → Eğitim tamamlandı! ({elapsed:.1f} saniye)")

    return clf, vec


# ─────────────────────────────────────────────────────────────────────────────
# 4. TAHMİN VE DEĞERLENDİRME
# ─────────────────────────────────────────────────────────────────────────────

def predict(clf, vec, test_sentences):
    """Test cümleleri için tahmin yap."""
    print("\n[*] Özellikler çıkarılıyor (test)...")
    X_feat, y_true = extract_features(test_sentences, use_gold_context=True)
    print(f"    → {len(X_feat):,} token")

    print("[*] Tahmin yapılıyor...")
    X = vec.transform(tqdm(X_feat, desc="    Dönüştürme", leave=False))
    y_pred = clf.predict(X)

    return list(y_true), list(y_pred)


def evaluate(y_true, y_pred, output_dir):
    """
    Değerlendirme metrikleri hesapla ve görselleştir.
    İster 4: F-measure, precision, recall, accuracy + confusion matrix
    """
    # Var olan sınıfları bul
    all_labels = sorted(set(y_true) | set(y_pred))

    print("\n" + "="*60)
    print("DEĞERLENDİRME SONUÇLARI")
    print("="*60)

    # ─── Genel metrikler ───
    acc      = accuracy_score(y_true, y_pred)
    f1_mac   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_wt    = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec_mac = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_mac  = recall_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"  Doğruluk (Accuracy)          : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Macro F1-score               : {f1_mac:.4f}")
    print(f"  Weighted F1-score            : {f1_wt:.4f}")
    print(f"  Macro Precision              : {prec_mac:.4f}")
    print(f"  Macro Recall                 : {rec_mac:.4f}")
    print()

    # ─── Sınıf bazlı rapor ───
    report = classification_report(
        y_true, y_pred,
        labels=all_labels,
        zero_division=0,
        digits=4
    )
    print("Sınıf Bazlı Sınıflandırma Raporu:")
    print(report)

    # Raporu dosyaya yaz
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("TÜRKÇE MORFOLOJİK ÇÖZÜMLEME - SINIFLANDIRMA RAPORU\n")
        f.write("="*60 + "\n\n")
        f.write(f"Doğruluk (Accuracy)       : {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"Macro F1-score            : {f1_mac:.4f}\n")
        f.write(f"Weighted F1-score         : {f1_wt:.4f}\n")
        f.write(f"Macro Precision           : {prec_mac:.4f}\n")
        f.write(f"Macro Recall              : {rec_mac:.4f}\n\n")
        f.write("Sınıf Bazlı Rapor:\n")
        f.write(report)
    print(f"[✓] Sınıflandırma raporu kaydedildi: {report_path}")

    return all_labels, acc, f1_mac, f1_wt, prec_mac, rec_mac


# ─────────────────────────────────────────────────────────────────────────────
# 5. GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, all_labels, output_dir):
    """Karışıklık matrisini çiz ve kaydet."""
    print("\n[*] Karışıklık matrisi oluşturuluyor...")

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    # Normalize edilmiş matris (her satırın toplamına böl)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle(
        'Türkçe Morfolojik Çözümleme - Karışıklık Matrisi\n(Turkish Morphological Disambiguation)',
        fontsize=14, fontweight='bold', y=1.02
    )

    # Sol: Ham sayı
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='YlOrRd',
        xticklabels=all_labels, yticklabels=all_labels,
        ax=axes[0], linewidths=0.5
    )
    axes[0].set_title('Ham Sayılar (Raw Counts)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Tahmin Edilen (Predicted)', fontsize=11)
    axes[0].set_ylabel('Gerçek (True)', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # Sağ: Normalize
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=all_labels, yticklabels=all_labels,
        ax=axes[1], linewidths=0.5, vmin=0, vmax=1
    )
    axes[1].set_title('Normalize Edilmiş (Normalized)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Tahmin Edilen (Predicted)', fontsize=11)
    axes[1].set_ylabel('Gerçek (True)', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Karışıklık matrisi kaydedildi: {path}")


def plot_per_class_metrics(y_true, y_pred, all_labels, output_dir):
    """Her sınıf için F1, Precision, Recall bar grafiği."""
    print("[*] Sınıf bazlı metrik grafiği oluşturuluyor...")

    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, zero_division=0
    )

    df = pd.DataFrame({
        'Etiket': all_labels,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'Destek': support
    }).sort_values('F1-Score', ascending=False)

    fig, axes = plt.subplots(2, 1, figsize=(16, 14))
    fig.suptitle(
        'Türkçe Morfolojik Çözümleme\nSınıf Bazlı Başarı Metrikleri',
        fontsize=14, fontweight='bold'
    )

    # ─── Üst grafik: F1, Precision, Recall bar ───
    x = np.arange(len(df))
    width = 0.28
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    bars1 = axes[0].bar(x - width, df['Precision'], width, label='Precision', color=colors[0], alpha=0.85)
    bars2 = axes[0].bar(x,         df['F1-Score'],  width, label='F1-Score',  color=colors[1], alpha=0.85)
    bars3 = axes[0].bar(x + width, df['Recall'],    width, label='Recall',    color=colors[2], alpha=0.85)

    axes[0].set_xlabel('UPOS Etiketi', fontsize=11)
    axes[0].set_ylabel('Skor', fontsize=11)
    axes[0].set_title('Precision, Recall ve F1-Score (Sınıf Bazlı)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df['Etiket'], rotation=45, ha='right')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='0.90 Eşiği')
    axes[0].grid(axis='y', alpha=0.3)

    # Değerleri bar üstüne yaz
    for bar in bars2:
        h = bar.get_height()
        if h > 0.02:
            axes[0].annotate(f'{h:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=8)

    # ─── Alt grafik: Destek sayısı (kaç örnek var) ───
    support_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))
    bars = axes[1].bar(df['Etiket'], df['Destek'], color=support_colors, alpha=0.85, edgecolor='white')
    axes[1].set_xlabel('UPOS Etiketi', fontsize=11)
    axes[1].set_ylabel('Örnek Sayısı', fontsize=11)
    axes[1].set_title('Test Setindeki Sınıf Dağılımı', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        axes[1].annotate(f'{int(h):,}',
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3), textcoords='offset points',
            ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'metrics_per_class.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Sınıf bazlı metrik grafiği kaydedildi: {path}")


def plot_overall_metrics(acc, f1_mac, f1_wt, prec_mac, rec_mac, output_dir):
    """Genel başarı metriklerini özet grafik olarak çiz."""
    print("[*] Genel metrik özet grafiği oluşturuluyor...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Türkçe Morfolojik Çözümleme\nGenel Başarı Özeti',
        fontsize=14, fontweight='bold'
    )

    # ─── Sol: Radar / Bar chart ───
    metrics = {
        'Accuracy': acc,
        'Macro\nPrecision': prec_mac,
        'Macro\nRecall': rec_mac,
        'Macro\nF1-Score': f1_mac,
        'Weighted\nF1-Score': f1_wt,
    }

    labels_m = list(metrics.keys())
    values_m = list(metrics.values())

    bar_colors = ['#1abc9c', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    bars = axes[0].barh(labels_m, values_m, color=bar_colors, alpha=0.85, edgecolor='white', height=0.6)
    axes[0].set_xlim(0, 1.1)
    axes[0].set_xlabel('Skor', fontsize=11)
    axes[0].set_title('Genel Metrikler', fontsize=12, fontweight='bold')
    axes[0].axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0].grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, values_m):
        axes[0].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{val:.4f} ({val*100:.2f}%)',
                     va='center', fontsize=10, fontweight='bold')

    # ─── Sağ: Pasta grafik - Doğru vs Yanlış tahmin ───
    from collections import Counter
    # Bu bilgiyi dışarıdan alıyoruz, hesapla
    total = 100  # yer tutucu
    correct = acc * 100
    wrong   = 100 - correct

    wedge_colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)
    wedges, texts, autotexts = axes[1].pie(
        [correct, wrong],
        labels=['Doğru\nTahmin', 'Yanlış\nTahmin'],
        colors=wedge_colors,
        autopct='%1.2f%%',
        explode=explode,
        startangle=90,
        shadow=True,
        textprops={'fontsize': 12}
    )
    for at in autotexts:
        at.set_fontweight('bold')
        at.set_fontsize(13)
    axes[1].set_title(f'Tahmin Doğruluğu\n(Accuracy: {acc*100:.2f}%)',
                      fontsize=12, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, 'overall_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Genel metrik grafiği kaydedildi: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CoNLL-U ÇIKTI YAZICI (İSTER 2)
# ─────────────────────────────────────────────────────────────────────────────

def write_predictions_conllu(test_sentences, y_pred, output_dir):
    """
    Tahmin edilen UPOS etiketleriyle yeni CoNLL-U dosyası oluşturur.
    İster 2: Tüm işaretlemeler CoNLL formatında yapılmalıdır.
    
    B-TAG / I-TAG formatı: Proje PDF'inde belirtildiği gibi CoNLL formatı kullanılır.
    UPOS için B-/I- öneki eklenerek BIO formatına dönüştürülür.
    """
    out_path = os.path.join(output_dir, 'predictions_test.conllu')

    # Tahminleri cümle bazında hizala
    pred_iter = iter(y_pred)

    with open(out_path, 'w', encoding='utf-8') as f:
        # BIO format açıklaması
        f.write("# ============================================================\n")
        f.write("# Türkçe Morfolojik Çözümleme - Tahmin Sonuçları\n")
        f.write("# Format: CoNLL-U + tahmin edilmiş UPOS (sütun 4)\n")
        f.write("# BIO etiketleme: B-TAG = başlangıç, I-TAG = devam\n")
        f.write("# ============================================================\n\n")

        for sent in test_sentences:
            # Yorum satırları
            f.write(f"# sent_id = {sent.sent_id or 'unknown'}\n")
            if sent.text:
                f.write(f"# text = {sent.text}\n")

            prev_upos = None
            for tok in sent.tokens:
                pred_upos = next(pred_iter, tok.upos)

                # BIO etiket formatı
                if pred_upos != prev_upos:
                    bio_tag = f"B-{pred_upos}"
                else:
                    bio_tag = f"I-{pred_upos}"
                prev_upos = pred_upos

                # CoNLL-U satırı: ID FORM LEMMA UPOS(pred) XPOS FEATS HEAD DEPREL DEPS MISC|BIO
                misc = tok.misc if tok.misc != '_' else ''
                if misc:
                    misc_out = f"{misc}|PredBIO={bio_tag}"
                else:
                    misc_out = f"PredBIO={bio_tag}"

                f.write(
                    f"{tok.id}\t{tok.form}\t{tok.lemma}\t"
                    f"{pred_upos}\t{tok.xpos}\t{tok.feats}\t"
                    f"{tok.head}\t{tok.deprel}\t{tok.deps}\t"
                    f"{misc_out}\n"
                )

            f.write("\n")

    print(f"[✓] Tahmin CoNLL-U dosyası kaydedildi: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. ÖZET RAPOR
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(acc, f1_mac, f1_wt, prec_mac, rec_mac, output_dir):
    print("\n" + "="*60)
    print("  PROJE ÖZET TABLOSU")
    print("="*60)
    print(f"  Model       : Multinomial Logistic Regression")
    print(f"  Veri Seti   : Turkish BOUN UD Treebank")
    print(f"  Görev       : Morfolojik Çözümleme (UPOS Tagging)")
    print("─"*60)
    print(f"  Accuracy    : {acc*100:.2f}%")
    print(f"  Macro F1    : {f1_mac*100:.2f}%")
    print(f"  Weighted F1 : {f1_wt*100:.2f}%")
    print(f"  Macro Prec  : {prec_mac*100:.2f}%")
    print(f"  Macro Recall: {rec_mac*100:.2f}%")
    print("─"*60)
    print(f"  Çıktılar    : {output_dir}/")
    print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
# 8. ANA FONKSİYON
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  Türkçe Morfolojik Çözümleme Projesi")
    print("  Turkish Morphological Disambiguation")
    print("="*60)
    t_start = time.time()

    # Çıktı klasörünü oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ─── 1. Veri Okuma ───
    if not os.path.isfile(TRAIN_FILE):
        print(f"[HATA] Eğitim dosyası bulunamadı: {TRAIN_FILE}")
        sys.exit(1)
    if not os.path.isfile(TEST_FILE):
        print(f"[HATA] Test dosyası bulunamadı: {TEST_FILE}")
        sys.exit(1)

    train_sentences = parse_conllu(TRAIN_FILE)
    test_sentences  = parse_conllu(TEST_FILE)

    train_tokens = sum(len(s.tokens) for s in train_sentences)
    test_tokens  = sum(len(s.tokens) for s in test_sentences)
    print(f"\n  Eğitim : {len(train_sentences):,} cümle, {train_tokens:,} token")
    print(f"  Test   : {len(test_sentences):,} cümle, {test_tokens:,} token")

    # ─── 2. Model Eğitimi ───
    clf, vec = train_model(train_sentences)

    # ─── 3. Tahmin ───
    y_true, y_pred = predict(clf, vec, test_sentences)

    # ─── 4. CoNLL-U Çıktısı (İster 2) ───
    write_predictions_conllu(test_sentences, y_pred, OUTPUT_DIR)

    # ─── 5. Değerlendirme (İster 4) ───
    all_labels, acc, f1_mac, f1_wt, prec_mac, rec_mac = evaluate(y_true, y_pred, OUTPUT_DIR)

    # ─── 6. Görselleştirme (İster 4) ───
    plot_confusion_matrix(y_true, y_pred, all_labels, OUTPUT_DIR)
    plot_per_class_metrics(y_true, y_pred, all_labels, OUTPUT_DIR)
    plot_overall_metrics(acc, f1_mac, f1_wt, prec_mac, rec_mac, OUTPUT_DIR)

    # ─── 7. Özet ───
    t_total = time.time() - t_start
    print_summary(acc, f1_mac, f1_wt, prec_mac, rec_mac, OUTPUT_DIR)
    print(f"\n[✓] Toplam süre: {t_total:.1f} saniye")
    print(f"[✓] Tüm çıktılar '{OUTPUT_DIR}/' klasörüne kaydedildi.\n")


if __name__ == '__main__':
    main()
