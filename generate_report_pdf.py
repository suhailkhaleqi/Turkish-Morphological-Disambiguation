"""
Türkçe Morfolojik Çözümleme Projesi — PDF Rapor Oluşturucu
Matplotlib PdfPages kullanarak profesyonel PDF raporu oluşturur.
Ekstra kütüphane gerekmez; matplotlib zaten kurulu.
"""

import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

OUTPUT_DIR = "output"
PDF_PATH   = "Rapor_Morfolojik_Cozumleme.pdf"

# ─── Renk paleti ───────────────────────────────────────────────
C_DARK   = "#0f172a"
C_NAVY   = "#1e3a8a"
C_BLUE   = "#2563eb"
C_VIOLET = "#7c3aed"
C_TEAL   = "#0d9488"
C_GREEN  = "#059669"
C_AMBER  = "#d97706"
C_RED    = "#dc2626"
C_LIGHT  = "#f8fafc"
C_BORDER = "#e2e8f0"
C_MUTED  = "#64748b"
C_WHITE  = "#ffffff"

# ─── Metrik verileri ───────────────────────────────────────────
METRICS_OVERALL = {
    "Accuracy":          0.9223,
    "Weighted F1":       0.9216,
    "Macro Precision":   0.8780,
    "Macro F1":          0.8482,
    "Macro Recall":      0.8332,
}

CLASS_DATA = [
    # (etiket, precision, recall, f1, support)
    ("PUNCT",  1.0000, 1.0000, 1.0000, 2028),
    ("DET",    0.9574, 0.9872, 0.9720,  546),
    ("CCONJ",  0.9459, 0.9347, 0.9403,  337),
    ("VERB",   0.9525, 0.9213, 0.9367, 2199),
    ("PRON",   0.9290, 0.9085, 0.9187,  317),
    ("NOUN",   0.9000, 0.9353, 0.9173, 3955),
    ("NUM",    0.9134, 0.9167, 0.9150,  276),
    ("ADP",    0.9609, 0.8371, 0.8947,  264),
    ("PART",   0.8298, 0.9630, 0.8914,  162),
    ("AUX",    0.8675, 0.9000, 0.8834,  240),
    ("PROPN",  0.8565, 0.8730, 0.8647,  677),
    ("ADV",    0.8455, 0.7979, 0.8210,  480),
    ("ADJ",    0.8448, 0.7753, 0.8086,  681),
    ("SCONJ",  0.8333, 0.3846, 0.5263,   26),
    ("INTJ",   0.5333, 0.3636, 0.4324,   22),
]

def set_style():
    plt.rcParams.update({
        'font.family':      'DejaVu Sans',
        'font.size':        10,
        'axes.spines.top':  False,
        'axes.spines.right':False,
        'figure.facecolor': C_LIGHT,
        'axes.facecolor':   C_WHITE,
        'text.color':       C_DARK,
        'axes.labelcolor':  C_DARK,
        'xtick.color':      C_MUTED,
        'ytick.color':      C_MUTED,
    })

def add_page_header(fig, title, subtitle="", page_num=None):
    """Her sayfanın üstüne başlık bandı çiz."""
    ax = fig.add_axes([0, 0.935, 1, 0.065])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    # Koyu arkaplan
    rect = FancyBboxPatch((0, 0), 1, 1,
                          boxstyle="square,pad=0",
                          facecolor=C_NAVY, edgecolor='none')
    ax.add_patch(rect)
    ax.text(0.02, 0.62, title, color=C_WHITE,
            fontsize=13, fontweight='bold', va='center', transform=ax.transAxes)
    if subtitle:
        ax.text(0.02, 0.22, subtitle, color='#93c5fd',
                fontsize=9, va='center', transform=ax.transAxes)
    if page_num:
        ax.text(0.98, 0.5, f"Sayfa {page_num}", color='#94a3b8',
                fontsize=8, ha='right', va='center', transform=ax.transAxes)

def add_page_footer(fig, text="Türkçe Morfolojik Çözümleme — NLP Dönem Projesi 2025-2026"):
    ax = fig.add_axes([0, 0, 1, 0.025])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.axhline(0.95, color=C_BORDER, linewidth=0.8)
    ax.text(0.5, 0.3, text, color=C_MUTED,
            fontsize=7.5, ha='center', va='center', transform=ax.transAxes)

def colored_box(ax, x, y, w, h, text, val_text,
                color=C_BLUE, text_color=C_WHITE):
    """Metrik kutusu çiz."""
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.01",
                          facecolor=color, edgecolor='none', alpha=0.92,
                          transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h*0.62, val_text, color=text_color,
            fontsize=18, fontweight='bold', ha='center', va='center',
            transform=ax.transAxes)
    ax.text(x + w/2, y + h*0.22, text, color=text_color,
            fontsize=8, ha='center', va='center', alpha=0.85,
            transform=ax.transAxes)


# ══════════════════════════════════════════════════════════════
#  SAYFA 1 — KAPAK
# ══════════════════════════════════════════════════════════════
def page_cover(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(C_DARK)

    # ── Üst dekoratif bölüm ──
    ax_top = fig.add_axes([0, 0.55, 1, 0.45])
    ax_top.set_xlim(0, 1); ax_top.set_ylim(0, 1); ax_top.axis('off')
    ax_top.set_facecolor(C_DARK)

    # Gradient benzeri arka plan daireler
    for r, alpha in [(0.6, 0.06), (0.45, 0.06), (0.30, 0.07)]:
        circ = plt.Circle((0.85, 1.1), r, color=C_VIOLET,
                           alpha=alpha, transform=ax_top.transAxes)
        ax_top.add_patch(circ)
    for r, alpha in [(0.5, 0.05), (0.35, 0.06)]:
        circ = plt.Circle((0.1, -0.1), r, color=C_BLUE,
                           alpha=alpha, transform=ax_top.transAxes)
        ax_top.add_patch(circ)

    # Rozet
    badge = FancyBboxPatch((0.05, 0.82), 0.52, 0.10,
                            boxstyle="round,pad=0.01",
                            facecolor="none",
                            edgecolor='#4b5563', linewidth=1,
                            transform=ax_top.transAxes)
    ax_top.add_patch(badge)
    ax_top.text(0.31, 0.87, "🎓  NLP Dönem Projesi — 2025–2026 Bahar",
                color='#94a3b8', fontsize=8.5, ha='center', va='center',
                transform=ax_top.transAxes)

    # Ana başlık
    ax_top.text(0.06, 0.65, "Türkçe", color='#93c5fd',
                fontsize=32, fontweight='bold', transform=ax_top.transAxes)
    ax_top.text(0.06, 0.44, "Morfolojik Çözümleme",
                color=C_WHITE, fontsize=26, fontweight='bold',
                transform=ax_top.transAxes)
    ax_top.text(0.06, 0.29, "Turkish Morphological Disambiguation",
                color='#64748b', fontsize=13, transform=ax_top.transAxes)

    # Alt çizgi
    ax_top.axhline(0.18, xmin=0.05, xmax=0.95,
                   color='#334155', linewidth=1)

    # Bilgi satırları
    info = [
        ("Model",     "Multinomial Logistic Regression"),
        ("Veri Seti", "Turkish BOUN UD Treebank"),
        ("Görev",     "POS Etiketleme (UPOS Tagging)"),
        ("Doğruluk",  "%92.23  |  Macro F1: %84.82"),
    ]
    for i, (lbl, val) in enumerate(info):
        yy = 0.12 - i * 0.032
        ax_top.text(0.07, yy, f"{lbl}:",
                    color='#64748b', fontsize=8.5,
                    transform=ax_top.transAxes)
        ax_top.text(0.22, yy, val,
                    color=C_WHITE, fontsize=8.5, fontweight='500',
                    transform=ax_top.transAxes)

    # ── Alt özet metrik bölümü ──
    ax_bot = fig.add_axes([0.05, 0.07, 0.90, 0.44])
    ax_bot.set_xlim(0, 1); ax_bot.set_ylim(0, 1); ax_bot.axis('off')
    ax_bot.set_facecolor(C_DARK)

    # Başlık
    ax_bot.text(0.5, 0.97, "Proje Başarı Özeti",
                color='#94a3b8', fontsize=9, ha='center',
                transform=ax_bot.transAxes)

    # 5 büyük metrik kutusu
    colors = [C_BLUE, C_TEAL, C_VIOLET, C_GREEN, "#b45309"]
    vals   = ["92.23%", "92.16%", "87.80%", "84.82%", "83.32%"]
    lbls   = ["Accuracy", "Weighted F1", "Macro Prec.", "Macro F1", "Macro Recall"]
    box_w  = 0.175
    gap    = 0.01
    start  = 0.01
    for i, (c, v, l) in enumerate(zip(colors, vals, lbls)):
        x = start + i*(box_w + gap)
        rect = FancyBboxPatch((x, 0.68), box_w, 0.25,
                              boxstyle="round,pad=0.015",
                              facecolor=c, edgecolor='none', alpha=0.88,
                              transform=ax_bot.transAxes)
        ax_bot.add_patch(rect)
        ax_bot.text(x + box_w/2, 0.835, v,
                    color=C_WHITE, fontsize=14.5, fontweight='bold',
                    ha='center', va='center', transform=ax_bot.transAxes)
        ax_bot.text(x + box_w/2, 0.71, l,
                    color='#cbd5e1', fontsize=7.5,
                    ha='center', va='center', transform=ax_bot.transAxes)

    # İster tablosu
    ister_data = [
        ("İster 1", "İstatistiksel ML Modeli (Multinomial LR)", "✓ Karşılandı"),
        ("İster 2", "CoNLL B/I Formatında İşaretleme",          "✓ Karşılandı"),
        ("İster 3", "Çalışır Eğitim ve Test Kodu",              "✓ Karşılandı"),
        ("İster 4", "F1 / Prec / Recall / Acc + Confusion Mat.","✓ Karşılandı"),
    ]
    col_colors = ['#1e293b', '#0f172a']
    for i, (ist, desc, stat) in enumerate(ister_data):
        yy = 0.56 - i * 0.12
        bg = FancyBboxPatch((0, yy - 0.03), 1.0, 0.10,
                            boxstyle="square,pad=0",
                            facecolor=col_colors[i % 2],
                            edgecolor='none',
                            transform=ax_bot.transAxes)
        ax_bot.add_patch(bg)
        ax_bot.text(0.01, yy + 0.02, ist,
                    color='#60a5fa', fontsize=8.5, fontweight='bold',
                    transform=ax_bot.transAxes)
        ax_bot.text(0.14, yy + 0.02, desc,
                    color='#e2e8f0', fontsize=8.5,
                    transform=ax_bot.transAxes)
        ax_bot.text(0.88, yy + 0.02, stat,
                    color='#4ade80', fontsize=8.5, fontweight='bold',
                    ha='right', transform=ax_bot.transAxes)

    # Teslim tarihi
    ax_bot.text(0.5, 0.03,
                "Teslim Tarihi: 12 Haziran 2026  |  Ders: Doğal Dil İşlemeye Giriş",
                color='#475569', fontsize=8, ha='center',
                transform=ax_bot.transAxes)

    pdf.savefig(fig, bbox_inches='tight', facecolor=C_DARK)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  SAYFA 2 — GİRİŞ + VERİ SETİ
# ══════════════════════════════════════════════════════════════
def page_intro(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(C_LIGHT)
    add_page_header(fig, "1. Giriş ve Veri Seti",
                    "Problem tanımı, veri istatistikleri ve CoNLL-U formatı", "2")
    add_page_footer(fig)

    ax = fig.add_axes([0.06, 0.08, 0.88, 0.84])
    ax.axis('off')
    ax.set_facecolor(C_LIGHT)

    # ── Giriş metni ──
    intro_lines = [
        ("1.1  Giriş ve Problem Tanımı", 0.960, 11, C_NAVY, 'bold'),
        ("", 0.940, 9, C_DARK, 'normal'),
        ("Morfolojik çözümleme (morphological disambiguation), doğal dil işlemenin temel", 0.926, 9, C_DARK, 'normal'),
        ("görevlerinden biridir. Türkçe gibi sondan eklemeli (agglutinative) dillerde bir sözcük", 0.912, 9, C_DARK, 'normal'),
        ("birden fazla morfolojik analizi bulunabilmektedir. Bu projede bağlam bilgisi kullanılarak", 0.898, 9, C_DARK, 'normal'),
        ("her sözcüğe en doğru Evrensel Sözcük Türü (UPOS) etiketi atanmaktadır.", 0.884, 9, C_DARK, 'normal'),
        ("", 0.868, 9, C_DARK, 'normal'),
        ("Türkçe, eklemeli yapısı nedeniyle POS etiketleme açısından zorlu bir dildir. Örneğin", 0.854, 9, C_DARK, 'normal'),
        ('"yaz" sözcüğü bağlama göre hem NOUN (yaz mevsimi) hem de VERB (yazmak fiili)', 0.840, 9, C_DARK, 'normal'),
        ("olarak etiketlenebilir. Bu belirsizliği çözmek için bağlam özellikleri kritik öneme sahiptir.", 0.826, 9, C_DARK, 'normal'),
    ]
    for (txt, y, fs, col, fw) in intro_lines:
        ax.text(0.0, y, txt, color=col, fontsize=fs, fontweight=fw,
                transform=ax.transAxes, va='top')

    # ── Veri seti başlık ──
    ax.text(0.0, 0.796, "1.2  Veri Seti — Turkish BOUN UD Treebank",
            color=C_NAVY, fontsize=11, fontweight='bold', transform=ax.transAxes)

    ax.text(0.0, 0.773,
            "Boğaziçi Üniversitesi tarafından oluşturulmuş, Universal Dependencies (UD) çerçevesindeki",
            color=C_DARK, fontsize=9, transform=ax.transAxes)
    ax.text(0.0, 0.759,
            "standart Türkçe dependency treebank'tir. CoNLL-U formatında saklanmaktadır.",
            color=C_DARK, fontsize=9, transform=ax.transAxes)

    # ── İstatistik kutuları ──
    stat_items = [
        ("9,761",   "Eğitim\nCümlesi",  C_BLUE),
        ("100,713", "Eğitim\nTokeni",   C_TEAL),
        ("~1,200",  "Test\nCümlesi",    C_VIOLET),
        ("12,210",  "Test\nTokeni",     C_GREEN),
        ("15",      "UPOS\nSınıfı",     C_AMBER),
    ]
    bw = 0.175; gap = 0.01; sx = 0.0
    for i, (val, lbl, col) in enumerate(stat_items):
        x = sx + i*(bw + gap)
        rect = FancyBboxPatch((x, 0.67), bw, 0.075,
                              boxstyle="round,pad=0.01",
                              facecolor=col, edgecolor='none', alpha=0.88,
                              transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + bw/2, 0.724, val,
                color=C_WHITE, fontsize=12, fontweight='bold',
                ha='center', va='center', transform=ax.transAxes)
        ax.text(x + bw/2, 0.685, lbl,
                color='#e2e8f0', fontsize=7, ha='center', va='center',
                transform=ax.transAxes)

    # ── CoNLL-U Format Açıklaması ──
    ax.text(0.0, 0.640, "1.3  CoNLL-U Dosya Formatı",
            color=C_NAVY, fontsize=11, fontweight='bold', transform=ax.transAxes)

    ax.text(0.0, 0.617, "Her token 10 sekme-ayrımlı (tab) sütundan oluşur:",
            color=C_DARK, fontsize=9, transform=ax.transAxes)

    # Tablo başlıkları
    headers = ["ID","FORM","LEMMA","UPOS","XPOS","FEATS","HEAD","DEPREL","DEPS","MISC"]
    widths  = [0.04, 0.09, 0.09, 0.07, 0.06, 0.22, 0.05, 0.09, 0.06, 0.23]
    y_tbl   = 0.594
    xs = [0.0]
    for w in widths[:-1]: xs.append(xs[-1]+w)

    # Header row
    hdr_bg = FancyBboxPatch((0.0, y_tbl - 0.003), 1.0, 0.024,
                             boxstyle="square,pad=0",
                             facecolor=C_NAVY, edgecolor='none',
                             transform=ax.transAxes)
    ax.add_patch(hdr_bg)
    for x_pos, hdr in zip(xs, headers):
        ax.text(x_pos + 0.005, y_tbl + 0.007, hdr,
                color=C_WHITE, fontsize=7, fontweight='bold',
                transform=ax.transAxes, va='center')

    # Veri satırları
    rows = [
        ["1","1936","1936","NUM","Year","Case=Nom|Num=Sing","2","nmod:poss","_","_"],
        ["2","yılında","yıl","NOUN","_","Case=Loc|Num=Sing","0","root","_","_"],
        ["3","yız","null","AUX","Zero","Number=Plur|Pers=1","2","cop","_","_"],
        ["4",".",".",  "PUNCT","Stop","_","2","punct","_","SpacesAfter=\\n"],
    ]
    row_colors = [C_WHITE, '#f8fafc', C_WHITE, '#f8fafc']
    for ri, (row, rc) in enumerate(zip(rows, row_colors)):
        yy = y_tbl - 0.022 - ri*0.022
        rbg = FancyBboxPatch((0.0, yy - 0.003), 1.0, 0.021,
                              boxstyle="square,pad=0",
                              facecolor=rc, edgecolor=C_BORDER, linewidth=0.3,
                              transform=ax.transAxes)
        ax.add_patch(rbg)
        for x_pos, cell in zip(xs, row):
            col = C_VIOLET if cell in ("NUM","NOUN","AUX","PUNCT") else C_DARK
            ax.text(x_pos + 0.005, yy + 0.007, cell,
                    color=col, fontsize=7,
                    transform=ax.transAxes, va='center',
                    fontweight='bold' if cell in ("NUM","NOUN","AUX","PUNCT") else 'normal')

    # ── UPOS Etiket Listesi ──
    y_upos = 0.46
    ax.text(0.0, y_upos, "1.4  Desteklenen UPOS Etiketleri (15 sınıf)",
            color=C_NAVY, fontsize=11, fontweight='bold', transform=ax.transAxes)

    tags = [
        ("NOUN",  C_BLUE,   "İsim"),
        ("VERB",  C_TEAL,   "Fiil"),
        ("ADJ",   C_VIOLET, "Sıfat"),
        ("ADV",   "#7c3aed","Zarf"),
        ("PROPN", "#0369a1","Özel İsim"),
        ("PUNCT", "#374151","Noktalama"),
        ("NUM",   C_AMBER,  "Sayı"),
        ("PRON",  "#065f46","Zamir"),
        ("DET",   "#1d4ed8","Belirteç"),
        ("AUX",   "#be185d","Yardımcı Fiil"),
        ("CCONJ", "#7e22ce","Bağlaç"),
        ("ADP",   "#c2410c","Edat"),
        ("PART",  "#4338ca","Parçacık"),
        ("SCONJ", "#92400e","Alt Cümle Bağ."),
        ("INTJ",  "#991b1b","Ünlem"),
    ]
    cols_per_row = 5
    tag_w = 0.19; tag_h = 0.052; tag_gap_x = 0.01; tag_gap_y = 0.006
    for ti, (tag, col, desc) in enumerate(tags):
        row_i = ti // cols_per_row
        col_i = ti % cols_per_row
        tx = col_i * (tag_w + tag_gap_x)
        ty = y_upos - 0.06 - row_i*(tag_h + tag_gap_y)
        rect = FancyBboxPatch((tx, ty), tag_w, tag_h,
                              boxstyle="round,pad=0.008",
                              facecolor=col, edgecolor='none', alpha=0.85,
                              transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(tx + tag_w/2, ty + tag_h*0.65, tag,
                color=C_WHITE, fontsize=9, fontweight='bold',
                ha='center', va='center', transform=ax.transAxes)
        ax.text(tx + tag_w/2, ty + tag_h*0.22, desc,
                color='#e2e8f0', fontsize=6.5, ha='center', va='center',
                transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight', facecolor=C_LIGHT)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  SAYFA 3 — YÖNTEM VE MODEL
# ══════════════════════════════════════════════════════════════
def page_method(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(C_LIGHT)
    add_page_header(fig, "2. Yöntem ve Model",
                    "Özellik mühendisliği, pipeline ve Multinomial Logistic Regression", "3")
    add_page_footer(fig)

    ax = fig.add_axes([0.06, 0.08, 0.88, 0.84])
    ax.axis('off'); ax.set_facecolor(C_LIGHT)

    # ── Pipeline ──
    ax.text(0.0, 0.960, "2.1  İşlem Akışı (Pipeline)",
            color=C_NAVY, fontsize=11, fontweight='bold', transform=ax.transAxes)

    steps = ["CoNLL-U\nOkuma","Özellik\nÇıkarımı","Vektörleştirme\n(DictVectorizer)",
             "Model\nEğitimi (LR)","Tahmin","Değerlendirme\n+ Grafikler"]
    step_colors = [C_TEAL, C_BLUE, C_VIOLET, C_NAVY, C_GREEN, C_AMBER]
    sw = 0.14; sh = 0.055; gap = 0.015; sy = 0.880
    sx_start = 0.0
    for i, (s, sc) in enumerate(zip(steps, step_colors)):
        x = sx_start + i*(sw + gap)
        if i > 0:
            ax.annotate('', xy=(x - 0.002, sy + sh/2),
                        xytext=(x - gap + 0.002, sy + sh/2),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color=C_MUTED, lw=1.5))
        rect = FancyBboxPatch((x, sy), sw, sh,
                              boxstyle="round,pad=0.008",
                              facecolor=sc, edgecolor='none', alpha=0.88,
                              transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + sw/2, sy + sh/2, s,
                color=C_WHITE, fontsize=7.5, fontweight='bold',
                ha='center', va='center', transform=ax.transAxes)

    # ── Özellik Grupları ──
    ax.text(0.0, 0.830, "2.2  Özellik Mühendisliği",
            color=C_NAVY, fontsize=11, fontweight='bold', transform=ax.transAxes)

    groups = [
        ("① Son-ek / Ön-ek Özellikleri", C_BLUE, [
            "suffix_1..5  →  Kelimenin son 1–5 karakteri (Türkçe ekler için kritik)",
            "prefix_1..3  →  Kelimenin ilk 1–3 karakteri",
        ]),
        ("② Morfolojik İpucu Özellikleri", C_VIOLET, [
            "has_progressive  →  -iyor/-ıyor eki (geniş zaman fiili → VERB)",
            "has_past_nfh     →  -mış/-miş eki (geçmiş zaman / sıfat fiil)",
            "has_past_fh      →  -dı/-di/-du/-dü eki (belirli geçmiş zaman)",
            "has_lik_suffix   →  -lık/-lik isim yapım eki → NOUN",
            "is_upper_start   →  Büyük harfle başlama → PROPN sinyali",
            "has_apostrophe   →  Kesme işareti (Türkiye'de gibi) → PROPN",
            "is_digit / has_digit  →  Sayı mı? → NUM etiketi",
        ]),
        ("③ Bağlam Özellikleri", C_TEAL, [
            "prev1_form, prev1_upos  →  Önceki tokenin formu ve UPOS etiketi",
            "prev2_form, prev2_upos  →  İki önceki tokenin bilgisi",
            "next1_form, next1_upos  →  Sonraki tokenin formu ve UPOS etiketi",
            "next2_form, next2_upos  →  İki sonraki tokenin bilgisi",
            "position_ratio_cat      →  Cümle içindeki konum (0–3 kategorik)",
        ]),
    ]

    y_cur = 0.808
    for (grp_title, grp_col, items) in groups:
        rect = FancyBboxPatch((0.0, y_cur - 0.003), 0.99, 0.023,
                              boxstyle="square,pad=0",
                              facecolor=grp_col, edgecolor='none', alpha=0.15,
                              transform=ax.transAxes)
        ax.add_patch(rect)
        circ_rect = FancyBboxPatch((0.0, y_cur - 0.003), 0.005, 0.023,
                                   boxstyle="square,pad=0",
                                   facecolor=grp_col, edgecolor='none',
                                   transform=ax.transAxes)
        ax.add_patch(circ_rect)
        ax.text(0.012, y_cur + 0.008, grp_title,
                color=grp_col, fontsize=9, fontweight='bold',
                transform=ax.transAxes)
        y_cur -= 0.025
        for item in items:
            ax.text(0.025, y_cur, f"•  {item}",
                    color=C_DARK, fontsize=8,
                    transform=ax.transAxes)
            y_cur -= 0.018
        y_cur -= 0.006

    # ── Model Bilgisi ──
    ax.text(0.0, y_cur - 0.005, "2.3  Multinomial Logistic Regression (İster 1)",
            color=C_NAVY, fontsize=11, fontweight='bold', transform=ax.transAxes)
    y_cur -= 0.030

    model_info = [
        ("solver='saga'",   "Büyük seyrek veri için hızlı çözücü"),
        ("max_iter=200",    "Maksimum iterasyon — yakınsama garantisi"),
        ("C=1.0",           "L2 regularization katsayısı (aşırı öğrenme engeli)"),
        ("n_jobs=-1",       "Tüm CPU çekirdekleri paralel kullanılır"),
        ("tol=0.01",        "Yakınsama toleransı"),
    ]
    col_w = [0.22, 0.55]
    # Tablo header
    hbg = FancyBboxPatch((0, y_cur - 0.002), 0.85, 0.020,
                          boxstyle="square,pad=0",
                          facecolor=C_NAVY, edgecolor='none',
                          transform=ax.transAxes)
    ax.add_patch(hbg)
    ax.text(0.01, y_cur + 0.006, "Parametre",
            color=C_WHITE, fontsize=8, fontweight='bold', transform=ax.transAxes)
    ax.text(0.23, y_cur + 0.006, "Açıklama",
            color=C_WHITE, fontsize=8, fontweight='bold', transform=ax.transAxes)
    y_cur -= 0.022

    for ri, (param, desc) in enumerate(model_info):
        rc = C_WHITE if ri % 2 == 0 else '#f1f5f9'
        rbg = FancyBboxPatch((0, y_cur - 0.002), 0.85, 0.019,
                              boxstyle="square,pad=0",
                              facecolor=rc, edgecolor=C_BORDER, linewidth=0.3,
                              transform=ax.transAxes)
        ax.add_patch(rbg)
        ax.text(0.01, y_cur + 0.006, param,
                color=C_VIOLET, fontsize=8, fontweight='bold',
                fontfamily='monospace', transform=ax.transAxes)
        ax.text(0.23, y_cur + 0.006, desc,
                color=C_DARK, fontsize=8, transform=ax.transAxes)
        y_cur -= 0.021

    # Süre kutuları
    y_cur -= 0.010
    for val, lbl, col in [("12.9 sn","Eğitim Süresi",C_BLUE),
                           ("34.7 sn","Toplam Süre",C_GREEN),
                           ("100,713","Eğitim Tokeni",C_VIOLET)]:
        rect = FancyBboxPatch((0, y_cur), 0.27, 0.042,
                              boxstyle="round,pad=0.008",
                              facecolor=col, edgecolor='none', alpha=0.85,
                              transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.135, y_cur + 0.028, val,
                color=C_WHITE, fontsize=11, fontweight='bold',
                ha='center', transform=ax.transAxes)
        ax.text(0.135, y_cur + 0.008, lbl,
                color='#e2e8f0', fontsize=7.5,
                ha='center', transform=ax.transAxes)
        # shift for next
        ax_next = ax.transAxes
        # manually offset
        rect2 = ax.transAxes
        # Use direct coordinate shift
        pass

    # Re-draw the three boxes properly side by side
    boxes = [
        (0.00, "12.9 sn", "Eğitim Süresi",  C_BLUE),
        (0.30, "34.7 sn", "Toplam Süre",     C_GREEN),
        (0.60, "100,713", "Eğitim Tokeni",   C_VIOLET),
    ]
    for (bx, val, lbl, col) in boxes:
        rect = FancyBboxPatch((bx, y_cur), 0.27, 0.042,
                              boxstyle="round,pad=0.008",
                              facecolor=col, edgecolor='none', alpha=0.85,
                              transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(bx + 0.135, y_cur + 0.028, val,
                color=C_WHITE, fontsize=11, fontweight='bold',
                ha='center', transform=ax.transAxes)
        ax.text(bx + 0.135, y_cur + 0.008, lbl,
                color='#e2e8f0', fontsize=7.5,
                ha='center', transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight', facecolor=C_LIGHT)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  SAYFA 4 — GENEL METRİKLER + PASTA GRAFİK
# ══════════════════════════════════════════════════════════════
def page_overall_metrics(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(C_LIGHT)
    add_page_header(fig, "3. Başarı Metrikleri",
                    "Genel metrikler, doğruluk özeti ve ağırlıklı F1 analizi", "4")
    add_page_footer(fig)

    # ── Büyük metrik kutuları ──
    ax_top = fig.add_axes([0.05, 0.80, 0.90, 0.12])
    ax_top.axis('off'); ax_top.set_facecolor(C_LIGHT)
    metrics_list = [
        ("Accuracy",     "92.23%", C_BLUE),
        ("Weighted F1",  "92.16%", C_TEAL),
        ("Macro Prec.",  "87.80%", C_VIOLET),
        ("Macro F1",     "84.82%", C_GREEN),
        ("Macro Recall", "83.32%", "#b45309"),
    ]
    bw = 0.18; bgap = 0.01
    for i, (lbl, val, col) in enumerate(metrics_list):
        x = i*(bw + bgap)
        rect = FancyBboxPatch((x, 0.05), bw, 0.88,
                              boxstyle="round,pad=0.02",
                              facecolor=col, edgecolor='none', alpha=0.88,
                              transform=ax_top.transAxes)
        ax_top.add_patch(rect)
        ax_top.text(x + bw/2, 0.62, val,
                    color=C_WHITE, fontsize=16, fontweight='bold',
                    ha='center', va='center', transform=ax_top.transAxes)
        ax_top.text(x + bw/2, 0.20, lbl,
                    color='#e2e8f0', fontsize=8, ha='center',
                    transform=ax_top.transAxes)

    # ── Yatay bar grafiği (Genel metrikler) ──
    ax_bar = fig.add_axes([0.05, 0.50, 0.56, 0.27])
    ax_bar.set_facecolor(C_WHITE)
    labels  = list(METRICS_OVERALL.keys())
    values  = list(METRICS_OVERALL.values())
    colors  = [C_BLUE, C_TEAL, C_VIOLET, C_GREEN, "#b45309"]
    bars = ax_bar.barh(labels, values, color=colors, alpha=0.85,
                       height=0.55, edgecolor='white', linewidth=0.5)
    ax_bar.set_xlim(0, 1.12)
    ax_bar.set_xlabel('Skor', fontsize=9, color=C_DARK)
    ax_bar.set_title('Genel Başarı Metrikleri', fontsize=10,
                     fontweight='bold', color=C_NAVY, pad=8)
    ax_bar.axvline(x=1.0, color=C_MUTED, linestyle='--', alpha=0.4, linewidth=1)
    ax_bar.tick_params(labelsize=8.5)
    for bar, val in zip(bars, values):
        ax_bar.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f} ({val*100:.2f}%)',
                    va='center', fontsize=8, fontweight='bold', color=C_DARK)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    # ── Pasta grafik ──
    ax_pie = fig.add_axes([0.63, 0.50, 0.34, 0.27])
    ax_pie.set_facecolor(C_WHITE)
    correct = 0.9223; wrong = 1 - correct
    wedges, texts, autotexts = ax_pie.pie(
        [correct, wrong],
        labels=['Doğru\nTahmin', 'Yanlış\nTahmin'],
        colors=[C_GREEN, C_RED],
        autopct='%1.2f%%',
        explode=(0.06, 0),
        startangle=90,
        shadow=False,
        textprops={'fontsize': 8},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_fontweight('bold')
        at.set_color(C_WHITE)
    ax_pie.set_title('Doğru / Yanlış\nTahmin Oranı',
                     fontsize=9, fontweight='bold', color=C_NAVY, pad=6)

    # ── Overall grafik PNG (var ise göster) ──
    overall_path = os.path.join(OUTPUT_DIR, 'overall_metrics.png')
    if os.path.exists(overall_path):
        ax_img = fig.add_axes([0.05, 0.17, 0.90, 0.30])
        img = plt.imread(overall_path)
        ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.set_title("Şekil 1 — Genel Başarı Özeti (Grafik)",
                         fontsize=9, color=C_MUTED, pad=4)

    # Açıklama notu
    ax_note = fig.add_axes([0.05, 0.09, 0.90, 0.07])
    ax_note.axis('off')
    note_bg = FancyBboxPatch((0, 0), 1, 1,
                              boxstyle="round,pad=0.02",
                              facecolor='#eff6ff', edgecolor='#bfdbfe',
                              linewidth=1, transform=ax_note.transAxes)
    ax_note.add_patch(note_bg)
    ax_note.text(0.01, 0.70, "ℹ  Weighted F1 vs Macro F1:",
                 color=C_BLUE, fontsize=8.5, fontweight='bold',
                 transform=ax_note.transAxes)
    ax_note.text(0.01, 0.30,
                 "Weighted F1 (%92.16), Macro F1'den (%84.82) yüksektir. Model sık sınıflarda (NOUN, VERB, PUNCT) çok "
                 "iyi performans gösterirken,",
                 color='#1e40af', fontsize=8, transform=ax_note.transAxes)

    pdf.savefig(fig, bbox_inches='tight', facecolor=C_LIGHT)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  SAYFA 5 — CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════
def page_confusion_matrix(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(C_LIGHT)
    add_page_header(fig, "4. Karışıklık Matrisi (Confusion Matrix)",
                    "Ham sayılar ve normalize edilmiş matris — her sınıf için hata analizi", "5")
    add_page_footer(fig)

    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    if os.path.exists(cm_path):
        ax_img = fig.add_axes([0.02, 0.30, 0.96, 0.62])
        img = plt.imread(cm_path)
        ax_img.imshow(img)
        ax_img.axis('off')

    # Açıklama
    ax_exp = fig.add_axes([0.05, 0.09, 0.90, 0.19])
    ax_exp.axis('off')

    ax_exp.text(0.0, 0.95, "4.1  Karışıklık Matrisi Yorumu",
                color=C_NAVY, fontsize=10, fontweight='bold',
                transform=ax_exp.transAxes)

    explanations = [
        ("Sol Matris (Ham Sayılar):",
         "Her satır gerçek etiketi, her sütun tahmin edilen etiketi gösterir. "
         "Köşegen üzerindeki değerler doğru tahminlerdir."),
        ("Sağ Matris (Normalize):",
         "Her satır kendi toplamına bölünmüştür. Değerler 0-1 arasında oransal olarak gösterilir."),
        ("Temel Gözlemler:",
         "PUNCT → Mükemmel (%100). ADJ↔NOUN karışıklığı: Türkçe'de sıfatlar isim gibi kullanılabilir. "
         "SCONJ↔ADV: Az örnekli SCONJ sınıfı yetersiz temsil edilmektedir."),
    ]
    y = 0.78
    for title, desc in explanations:
        ax_exp.text(0.01, y, f"• {title}", color=C_BLUE, fontsize=8.5,
                    fontweight='bold', transform=ax_exp.transAxes)
        y -= 0.16
        ax_exp.text(0.04, y, desc, color=C_DARK, fontsize=8,
                    transform=ax_exp.transAxes, wrap=True)
        y -= 0.20

    pdf.savefig(fig, bbox_inches='tight', facecolor=C_LIGHT)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  SAYFA 6 — SINIF BAZLI METRİKLER (Grafik)
# ══════════════════════════════════════════════════════════════
def page_class_metrics_chart(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(C_LIGHT)
    add_page_header(fig, "5. Sınıf Bazlı Metrik Grafiği",
                    "Her UPOS etiketi için Precision, Recall, F1 ve sınıf dağılımı", "6")
    add_page_footer(fig)

    per_class_path = os.path.join(OUTPUT_DIR, 'metrics_per_class.png')
    if os.path.exists(per_class_path):
        ax_img = fig.add_axes([0.02, 0.09, 0.96, 0.84])
        img = plt.imread(per_class_path)
        ax_img.imshow(img)
        ax_img.axis('off')

    pdf.savefig(fig, bbox_inches='tight', facecolor=C_LIGHT)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  SAYFA 7 — SINIF BAZLI TABLO
# ══════════════════════════════════════════════════════════════
def page_class_table(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(C_LIGHT)
    add_page_header(fig, "6. Sınıf Bazlı Detay Tablosu",
                    "15 UPOS etiketi için Precision, Recall, F1-Score ve destek sayısı", "7")
    add_page_footer(fig)

    ax = fig.add_axes([0.05, 0.08, 0.90, 0.84])
    ax.axis('off'); ax.set_facecolor(C_LIGHT)

    ax.text(0.0, 0.96, "6.1  Her UPOS Sınıfı İçin Başarı Sonuçları",
            color=C_NAVY, fontsize=11, fontweight='bold', transform=ax.transAxes)
    ax.text(0.0, 0.938,
            "Aşağıdaki tablo 12,210 tokenlik test seti üzerinde hesaplanmıştır. F1-Score'a göre azalan sırada.",
            color=C_DARK, fontsize=8.5, transform=ax.transAxes)

    # Tablo çiz
    col_headers = ["UPOS", "Precision", "Recall", "F1-Score", "Destek", "Durum"]
    col_x       = [0.00,    0.14,        0.28,     0.42,       0.58,     0.70]
    col_w_total = 0.99
    tbl_y_start = 0.910

    # Başlık satırı
    hbg = FancyBboxPatch((0, tbl_y_start - 0.004), col_w_total, 0.028,
                          boxstyle="square,pad=0",
                          facecolor=C_NAVY, edgecolor='none',
                          transform=ax.transAxes)
    ax.add_patch(hbg)
    for x, hdr in zip(col_x, col_headers):
        ax.text(x + 0.005, tbl_y_start + 0.008, hdr,
                color=C_WHITE, fontsize=8.5, fontweight='bold',
                transform=ax.transAxes)

    # Renk eşleştirici
    def status_color(f1):
        if f1 >= 0.95: return ("#d1fae5", "#065f46", "Mükemmel")
        if f1 >= 0.90: return ("#dbeafe", "#1e40af", "Çok İyi")
        if f1 >= 0.85: return ("#ede9fe", "#5b21b6", "İyi")
        if f1 >= 0.70: return ("#fef3c7", "#92400e", "Orta")
        return ("#fee2e2", "#991b1b", "Zayıf ⚠")

    row_h = 0.036
    y_cur = tbl_y_start - 0.034
    for ri, (tag, prec, rec, f1, sup) in enumerate(CLASS_DATA):
        rc = C_WHITE if ri % 2 == 0 else '#f8fafc'
        rbg = FancyBboxPatch((0, y_cur - 0.004), col_w_total, row_h,
                              boxstyle="square,pad=0",
                              facecolor=rc, edgecolor=C_BORDER, linewidth=0.3,
                              transform=ax.transAxes)
        ax.add_patch(rbg)

        # F1 progress bar
        bar_x = col_x[3] + 0.005
        bar_w_max = 0.135
        bar_h = 0.010
        bar_y = y_cur + row_h/2 - bar_h/2

        bar_col = C_GREEN if f1 >= 0.90 else (C_BLUE if f1 >= 0.80 else C_AMBER if f1 >= 0.60 else C_RED)
        bar_bg = FancyBboxPatch((bar_x, bar_y), bar_w_max, bar_h,
                                 boxstyle="square,pad=0",
                                 facecolor='#e2e8f0', edgecolor='none',
                                 transform=ax.transAxes)
        ax.add_patch(bar_bg)
        bar_fill = FancyBboxPatch((bar_x, bar_y), bar_w_max * f1, bar_h,
                                   boxstyle="square,pad=0",
                                   facecolor=bar_col, edgecolor='none', alpha=0.8,
                                   transform=ax.transAxes)
        ax.add_patch(bar_fill)

        # Durum rozeti
        st_bg, st_txt, st_lbl = status_color(f1)
        st_x = col_x[5]
        srect = FancyBboxPatch((st_x, y_cur + row_h*0.15), 0.28, row_h*0.65,
                                boxstyle="round,pad=0.005",
                                facecolor=st_bg, edgecolor='none',
                                transform=ax.transAxes)
        ax.add_patch(srect)
        ax.text(st_x + 0.14, y_cur + row_h/2, st_lbl,
                color=st_txt, fontsize=7.5, fontweight='bold',
                ha='center', va='center', transform=ax.transAxes)

        # Değerler
        vals = [tag, f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{sup:,}"]
        for ci, (cx, val) in enumerate(zip(col_x, vals)):
            fc = C_VIOLET if ci == 0 else C_DARK
            fw = 'bold' if ci == 0 else 'normal'
            ax.text(cx + 0.005, y_cur + row_h/2, val,
                    color=fc, fontsize=8, fontweight=fw,
                    va='center', transform=ax.transAxes,
                    fontfamily='monospace' if ci > 0 else 'sans-serif')

        y_cur -= row_h

    # Genel ortalamalar
    y_cur -= 0.005
    avg_bg = FancyBboxPatch((0, y_cur - 0.004), col_w_total, row_h,
                             boxstyle="square,pad=0",
                             facecolor='#1e3a8a', edgecolor='none',
                             transform=ax.transAxes)
    ax.add_patch(avg_bg)
    avg_vals = ["ORTALAMA", "0.8780", "0.8332", "0.8482", "12,210", "—"]
    for ci, (cx, val) in enumerate(zip(col_x, avg_vals)):
        ax.text(cx + 0.005, y_cur + row_h/2, val,
                color=C_WHITE, fontsize=8, fontweight='bold',
                va='center', transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight', facecolor=C_LIGHT)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  SAYFA 8 — CoNLL ÇIKTI + TARTIŞMA + SONUÇ
# ══════════════════════════════════════════════════════════════
def page_conclusion(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(C_LIGHT)
    add_page_header(fig, "7. CoNLL Çıktı, Tartışma ve Sonuç",
                    "BIO etiketleme, hata analizi ve proje değerlendirmesi", "8")
    add_page_footer(fig)

    ax = fig.add_axes([0.05, 0.08, 0.88, 0.84])
    ax.axis('off'); ax.set_facecolor(C_LIGHT)

    # ── CoNLL çıktı ──
    ax.text(0.0, 0.960, "7.1  CoNLL-U Çıktı Formatı (İster 2)",
            color=C_NAVY, fontsize=11, fontweight='bold', transform=ax.transAxes)
    ax.text(0.0, 0.935,
            "predictions_test.conllu dosyasında 12,210 token BIO etiketli CoNLL-U formatında yazılmıştır:",
            color=C_DARK, fontsize=8.5, transform=ax.transAxes)

    # Kod kutusu
    code_bg = FancyBboxPatch((0, 0.80), 0.99, 0.128,
                              boxstyle="round,pad=0.01",
                              facecolor='#0f172a', edgecolor='#334155',
                              linewidth=1, transform=ax.transAxes)
    ax.add_patch(code_bg)

    code_lines = [
        ("# sent_id = boun_test_001", "#64748b"),
        ("# text = Kitabı çok beğendim.", "#64748b"),
        ("1  Kitabı   kitap  NOUN  _  Case=Acc  2  obj    _  PredBIO=B-NOUN", "#e2e8f0"),
        ("2  çok      çok    ADV   _  _         3  advmod _  PredBIO=B-ADV", "#e2e8f0"),
        ("3  beğendim beğen  VERB  _  Tense=Past 0 root   _  PredBIO=B-VERB", "#e2e8f0"),
        ("4  .        .      PUNCT _  _          3 punct  _  PredBIO=B-PUNCT", "#e2e8f0"),
    ]
    for li, (cline, col) in enumerate(code_lines):
        ax.text(0.015, 0.908 - li*0.017, cline,
                color=col, fontsize=7.5, fontfamily='monospace',
                transform=ax.transAxes)

    # BIO açıklaması
    bio_items = [
        ("B-TAG", C_GREEN, "Etiket grubunun BAŞLANGICI (B: Beginning)"),
        ("I-TAG", "#facc15","Aynı etiket grubunun DEVAMI (I: Inside/Continue)"),
    ]
    y_bio = 0.788
    for tag, col, desc in bio_items:
        trect = FancyBboxPatch((0.0, y_bio - 0.003), 0.10, 0.022,
                                boxstyle="round,pad=0.005",
                                facecolor=col, edgecolor='none', alpha=0.85,
                                transform=ax.transAxes)
        ax.add_patch(trect)
        ax.text(0.05, y_bio + 0.007, tag,
                color=C_WHITE if col != "#facc15" else C_DARK,
                fontsize=8.5, fontweight='bold', ha='center',
                transform=ax.transAxes)
        ax.text(0.115, y_bio + 0.007, desc,
                color=C_DARK, fontsize=8, transform=ax.transAxes)
        y_bio -= 0.028

    # ── Tartışma ──
    ax.text(0.0, 0.720, "7.2  Tartışma",
            color=C_NAVY, fontsize=11, fontweight='bold', transform=ax.transAxes)

    strong_pts = [
        ("PUNCT %100.00",    "is_punct özelliği sayesinde mükemmel tanıma"),
        ("VERB  %93.67",     "-iyor/-dı/-mış son-ekleri güçlü VERB sinyali"),
        ("NOUN  %91.73",     "En sık sınıf; yüksek destek ile iyi öğrenilmiş"),
        ("Hız   12.9 sn",    "100K+ token verisi çok hızlı eğitildi"),
    ]
    weak_pts = [
        ("INTJ  %43.24",    "Sadece 22 örnek — yetersiz veri"),
        ("SCONJ %52.63",    "Sadece 26 örnek + ADV ile karışıklık"),
        ("ADJ   %80.86",    "NOUN ile morfolojik örtüşme var"),
    ]

    ax.text(0.0, 0.695, "Güçlü Yönler:", color=C_GREEN,
            fontsize=9, fontweight='bold', transform=ax.transAxes)
    y = 0.672
    for title, desc in strong_pts:
        ax.text(0.02, y, f"✓  {title:18s}  {desc}",
                color=C_DARK, fontsize=8.2, transform=ax.transAxes)
        y -= 0.022

    ax.text(0.0, y - 0.008, "Zayıf Yönler / İyileştirme Önerileri:", color=C_RED,
            fontsize=9, fontweight='bold', transform=ax.transAxes)
    y -= 0.030
    for title, desc in weak_pts:
        ax.text(0.02, y, f"⚠  {title:18s}  {desc}",
                color=C_DARK, fontsize=8.2, transform=ax.transAxes)
        y -= 0.022

    # ── Sonuç kutusu ──
    y -= 0.015
    rect_sonuc = FancyBboxPatch((0.0, y - 0.13), 0.99, 0.145,
                                 boxstyle="round,pad=0.015",
                                 facecolor='#f0fdf4', edgecolor='#bbf7d0',
                                 linewidth=1.5, transform=ax.transAxes)
    ax.add_patch(rect_sonuc)
    ax.text(0.5, y - 0.005, "7.3  Sonuç",
            color=C_GREEN, fontsize=11, fontweight='bold',
            ha='center', transform=ax.transAxes)
    concl = (
        "Bu projede Türkçe morfolojik çözümleme görevi Multinomial Logistic Regression algoritması\n"
        "ve zengin özellik mühendisliği ile başarıyla gerçekleştirilmiştir. 12,210 tokenlik test\n"
        "setinde elde edilen %92.23 doğruluk ve %84.82 Macro F1 skoru, Türkçe gibi morfolojik\n"
        "açıdan karmaşık bir dil için tatmin edici bir başarı düzeyini ifade etmektedir.\n\n"
        "Proje 4 isterin tamamını karşılamaktadır: (1) Multinomial LR modeli,\n"
        "(2) CoNLL B/I formatı, (3) çalışır eğitim+test kodu, (4) grafikler ve metrikler."
    )
    ax.text(0.02, y - 0.025, concl,
            color='#065f46', fontsize=8.2, transform=ax.transAxes,
            va='top', linespacing=1.5)

    # İster kontrol listesi
    y2 = y - 0.165
    for ist, stat in [("İster 1 — Multinomial Logistic Regression kullanıldı","✓"),
                      ("İster 2 — CoNLL B/I formatında 12,210 token işaretlendi","✓"),
                      ("İster 3 — Tek komutla çalışan eğitim+test pipeline","✓"),
                      ("İster 4 — F1/Prec/Recall/Acc + 3 grafik üretildi","✓")]:
        ax.text(0.01, y2, f"{stat}  {ist}",
                color=C_GREEN if stat == "✓" else C_RED,
                fontsize=8.5, fontweight='bold' if stat == "✓" else 'normal',
                transform=ax.transAxes)
        y2 -= 0.025

    pdf.savefig(fig, bbox_inches='tight', facecolor=C_LIGHT)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    set_style()
    print(f"[*] PDF rapor oluşturuluyor: {PDF_PATH}")

    with PdfPages(PDF_PATH) as pdf:
        # Metadata
        d = pdf.infodict()
        d['Title']   = 'Türkçe Morfolojik Çözümleme — NLP Dönem Projesi'
        d['Author']  = 'NLP Dönem Projesi 2025-2026'
        d['Subject'] = 'Multinomial Logistic Regression ile UPOS Etiketleme'
        d['Keywords']= 'NLP, POS Tagging, Turkish, Morphological Disambiguation'

        print("  [1/8] Kapak sayfası...")
        page_cover(pdf)

        print("  [2/8] Giriş ve veri seti...")
        page_intro(pdf)

        print("  [3/8] Yöntem ve model...")
        page_method(pdf)

        print("  [4/8] Genel metrikler...")
        page_overall_metrics(pdf)

        print("  [5/8] Karışıklık matrisi...")
        page_confusion_matrix(pdf)

        print("  [6/8] Sınıf bazlı metrik grafiği...")
        page_class_metrics_chart(pdf)

        print("  [7/8] Sınıf bazlı tablo...")
        page_class_table(pdf)

        print("  [8/8] CoNLL çıktı, tartışma ve sonuç...")
        page_conclusion(pdf)

    print(f"\n[✓] PDF oluşturuldu: {PDF_PATH}")
    print(f"[✓] Toplam 8 sayfa")


if __name__ == '__main__':
    main()
