import os
from src.training.reproducibility import repository_root

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Arial", 8)
        self.setFillColor(colors.HexColor('#64748B'))
        
        # Header (On page 2 and later)
        if self._pageNumber > 1:
            self.drawString(36, 810, "Polypharmacy AI - ChemBERTa & Cross-Attention Hiyerarşik Mimari Raporu")
            self.setStrokeColor(colors.HexColor('#CBD5E1'))
            self.setLineWidth(0.5)
            self.line(36, 804, 559, 804)
            
        # Footer (On all pages)
        self.setStrokeColor(colors.HexColor('#CBD5E1'))
        self.setLineWidth(0.5)
        self.line(36, 42, 559, 42)
        
        self.drawString(36, 30, "Antigravity AI Lab | ChemBERTa + Multi-Head Cross-Attention Mimarisi | Gizli & Teknik Döküman")
        page_str = f"Sayfa {self._pageNumber} / {page_count}"
        self.drawRightString(559, 30, page_str)
        self.restoreState()

def generate_architecture_report_pdf():
    pdf_path = str(repository_root()) + "/Polypharmacy_AI_Mimari_Raporu.pdf"
    
    # Register TrueType Fonts
    pdfmetrics.registerFont(TTFont('Arial', '/System/Library/Fonts/Supplemental/Arial.ttf'))
    pdfmetrics.registerFont(TTFont('Arial-Bold', '/System/Library/Fonts/Supplemental/Arial Bold.ttf'))
    pdfmetrics.registerFont(TTFont('Arial-Italic', '/System/Library/Fonts/Supplemental/Arial Italic.ttf'))
    pdfmetrics.registerFont(TTFont('Arial-BoldItalic', '/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf'))
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=50
    )
    
    styles = getSampleStyleSheet()
    
    # Palette
    c_primary = colors.HexColor('#0F172A')    # Slate 900
    c_accent = colors.HexColor('#1E40AF')     # Dark Blue
    c_teal = colors.HexColor('#0F766E')       # Dark Teal
    c_card_bg = colors.HexColor('#F8FAFC')    # Slate 50
    c_border = colors.HexColor('#E2E8F0')     # Slate 200
    c_text = colors.HexColor('#334155')       # Slate 700
    c_header_bg = colors.HexColor('#F1F5F9')  # Slate 100
    
    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        fontName='Arial-Bold',
        fontSize=18,
        leading=22,
        textColor=c_primary,
        alignment=0
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        fontName='Arial',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#475569')
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        fontName='Arial-Bold',
        fontSize=12,
        leading=15,
        textColor=c_accent,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        fontName='Arial',
        fontSize=8.2,
        leading=11.5,
        textColor=c_text
    )
    
    bold_style = ParagraphStyle(
        'Bold_Custom',
        fontName='Arial-Bold',
        fontSize=8.2,
        leading=11.5,
        textColor=c_primary
    )
    
    code_style = ParagraphStyle(
        'Code_Custom',
        fontName='Arial',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#0F172A')
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        fontName='Arial',
        fontSize=7.8,
        leading=10.2,
        textColor=c_text
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        fontName='Arial-Bold',
        fontSize=7.8,
        leading=10.2,
        textColor=c_primary
    )
    
    story = []
    
    # ==========================================
    # HEADER SECTION
    # ==========================================
    header_data = [
        [
            Paragraph("<b>POLİFARMASİ YAN ETKİ TAHMİNİ İÇİN HİYERARŞİK GRAF SİNİR AĞI MİMARİSİ</b>", title_style),
        ],
        [
            Paragraph("<b>ChemBERTa Transformer + Multi-Head Cross-Attention + Log-Odds Füzyon Raporu</b> | Ağustos 2026", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[523])
    header_table.setStyle(TableStyle([
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.8, color=c_accent, spaceBefore=4, spaceAfter=6))
    
    # Executive Summary Card
    exec_text = (
        "<b>📌 YÖNETİCİ ÖZETİ (EXECUTIVE SUMMARY):</b><br/>"
        "Bu rapor, çoklu ilaç etkileşimlerini (polifarmasi) tahmin etmek amacıyla geliştirilen ve son aşamada "
        "<b>ChemBERTa-77M Moleküler Dil Modeli</b>, <b>Çok Başlı Çapraz Dikkat (Multi-Head Cross-Attention)</b> ve "
        "<b>Log-Odds Toplamsal Füzyon</b> bileşenleriyle güçlendirilen en gelişmiş yapay zeka mimarisini sunmaktadır. "
        "Sistemimiz <b>2.42 Milyon Biyoaktif Molekül</b> ve <b>9.61 Milyon Biyolojik Bağlantı</b> ile eğitilmiştir. "
        "Model <b>%89.52 Organ Sistemi Doğruluğu</b>, <b>%96.18 Organ AUROC</b> ve <b>%89.04 Spesifik AUROC</b> skorlarına ulaşmıştır."
    )
    exec_table = Table([[Paragraph(exec_text, body_style)]], colWidths=[523])
    exec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#EFF6FF')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#93C5FD')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(exec_table)
    story.append(Spacer(1, 5))
    
    # ==========================================
    # SECTION 1: UYGULANAN 3 BÜYÜK YENİLİK
    # ==========================================
    story.append(Paragraph("1. Uygulanan 3 Büyük Mimari Yenilik", h1_style))
    
    inv_text = (
        "• <b>🧪 1. ChemBERTa-77M Transformer Temsilleri:</b> 10 Milyon kimyasal bileşik üzerinde eğitilmiş RoBERTa tabanlı "
        "moleküler dil modelinden her ilacın 384 boyutlu yoğun kimyasal zekası çıkarıldı ve toplam nitelik vektörü <b>3,248 boyuta</b> genişletildi.<br/>"
        "• <b>🔄 2. Multi-Head Cross-Attention Mekanizması:</b> İlaç A ve İlaç B arasındaki moleküler çapraz etkileşim "
        "çift yönlü dikkat mekanizmasıyla (<i>A attends to B, B attends to A</i>) modellendi. Hangi kimyasal halkaların birbiriyle çakıştığı öğrenildi.<br/>"
        "• <b>➕ 3. Log-Odds Toplamsal Füzyon (Additive Fusion):</b> Olasılık çarpımı yerine Log-Odds uzayında toplama yapılarak "
        "olasılıkların aşağı büzülmesi engellendi; Seviye 1 organ başarısı doğrudan Seviye 2'ye aktarıldı."
    )
    inv_table = Table([[Paragraph(inv_text, body_style)]], colWidths=[523])
    inv_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_card_bg),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(inv_table)
    story.append(Spacer(1, 5))
    
    # ==========================================
    # SECTION 2: VERİ VE NİTELİK MİMARİSİ
    # ==========================================
    story.append(Paragraph("2. Zenginleştirilmiş Veri ve Nitelik Havuzu (3,248 Boyut)", h1_style))
    
    data_matrix = [
        [Paragraph("<b>Veri / Nitelik Katmanı</b>", table_cell_bold), Paragraph("<b>Boyut / Varlık Hacmi</b>", table_cell_bold), Paragraph("<b>Açıklama & Fonksiyon</b>", table_cell_bold)],
        [Paragraph("<b>ChemBERTa-77M Transformer</b>", table_cell), Paragraph("384 Boyut", table_cell), Paragraph("10M molekülden öğrenilmiş derin kimyasal zeka", table_cell)],
        [Paragraph("<b>Morgan (ECFP4) + MACCS</b>", table_cell), Paragraph("679 Boyut (512 + 167)", table_cell), Paragraph("Atomik fonksiyonel gruplar ve alt yapılar", table_cell)],
        [Paragraph("<b>RDKit Topological + Descriptors</b>", table_cell), Paragraph("522 Boyut (512 + 10)", table_cell), Paragraph("Atom yolları, MW, LogP, TPSA, H-Bağları", table_cell)],
        [Paragraph("<b>Hastalık Endikasyon Vektörü</b>", table_cell), Paragraph("1,363 Boyut", table_cell), Paragraph("İlacın tedavi ettiği hastalıklar (BioSNAP)", table_cell)],
        [Paragraph("<b>DTI Protein Hedef Ağı</b>", table_cell), Paragraph("300 Boyut", table_cell), Paragraph("İlacın bağlandığı 300 insan hedef proteini", table_cell)],
        [Paragraph("<b>BİRLEŞİK İLAÇ VEKTÖRÜ</b>", table_cell_bold), Paragraph("<b>3,248 BOYUT</b>", table_cell_bold), Paragraph("<b>Uçtan Uca Biyomedikal Temsil Matrisi</b>", table_cell_bold)]
    ]
    data_table = Table(data_matrix, colWidths=[160, 120, 243])
    data_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_header_bg),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#F8FAFC')),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(data_table)
    story.append(Spacer(1, 5))
    
    # ==========================================
    # SECTION 3: MODEL BAŞARIM METRİKLERİ
    # ==========================================
    story.append(Paragraph("3. Güncel Model Başarım Raporu ve İterasyon Karşılaştırması", h1_style))
    
    results_matrix = [
        [Paragraph("<b>Model Mimarisi</b>", table_cell_bold), Paragraph("<b>Organ Doğruluk</b>", table_cell_bold), Paragraph("<b>Organ AUROC</b>", table_cell_bold), Paragraph("<b>Spesifik AUROC</b>", table_cell_bold), Paragraph("<b>Spesifik AUPRC</b>", table_cell_bold), Paragraph("<b>Süre</b>", table_cell_bold)],
        [Paragraph("1. Logistic Regression Baseline", table_cell), Paragraph("%74.12", table_cell), Paragraph("%81.40", table_cell), Paragraph("%76.80", table_cell), Paragraph("0.4602", table_cell), Paragraph("~9 dk", table_cell)],
        [Paragraph("2. Standart Decagon GNN", table_cell), Paragraph("%85.09", table_cell), Paragraph("%88.71", table_cell), Paragraph("%86.20", table_cell), Paragraph("0.5309", table_cell), Paragraph("25 sn", table_cell)],
        [Paragraph("3. Unified Hierarchical GNN", table_cell), Paragraph("%89.26", table_cell), Paragraph("%96.02", table_cell), Paragraph("%88.52", table_cell), Paragraph("0.5175", table_cell), Paragraph("67 sn", table_cell)],
        [Paragraph("<b>4. ChemBERTa + Cross-Attn AI</b>", table_cell_bold), Paragraph("<b>%89.52</b>", table_cell_bold), Paragraph("<b>%96.18</b>", table_cell_bold), Paragraph("<b>%89.04</b>", table_cell_bold), Paragraph("<b>0.5302</b>", table_cell_bold), Paragraph("<b>148 sn</b>", table_cell_bold)]
    ]
    results_table = Table(results_matrix, colWidths=[155, 75, 75, 75, 75, 68])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_header_bg),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#F0FDF4')),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 5))
    
    # ==========================================
    # SECTION 4: 2 SEVİYELİ KARAR REHBERİ
    # ==========================================
    story.append(Paragraph("4. 2 Seviyeli Karar Mekanizması ve Klinik Simülasyon", h1_style))
    
    l1_desc = (
        "<b>🏛️ SEVİYE 1: ORGAN SİSTEMİ RİSKİ</b><br/>"
        "<b>%89.52 Doğruluk | %96.18 AUROC | 0.9243 AUPRC</b><br/>"
        "15 MedDRA Organ Sistemi: Kalp-Damar, Karaciğer, Mide-Bağırsak, Böbrek, Beyin, Solunum, Kan, Deri, vb.<br/>"
        "<i>Doktoru doğrudan tehlikedeki organa odaklar.</i>"
    )
    l2_desc = (
        "<b>🔬 SEVİYE 2: SPESİFİK YAN ETKİ TEŞHİSİ</b><br/>"
        "<b>%89.04 AUROC | SIDER 4.1 Priors | P@5: 0.2624</b><br/>"
        "100 Klinik Teşhis: Aritmi, Sarılık, Mide Kanaması, Akut Böbrek Hasarı, Trombositopeni, vb.<br/>"
        "<i>Cross-Attention ile çakışan atom halkalarını puanlar.</i>"
    )
    spec_table = Table([[Paragraph(l1_desc, body_style), Paragraph(l2_desc, body_style)]], colWidths=[256, 257])
    spec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_card_bg),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(spec_table)
    story.append(Spacer(1, 5))
    
    # Bottom Paths Box
    files_text = (
        "<b>📁 ÜRETİLEN ARTIFACT VE MODEL DOSYALARI:</b><br/>"
        "• <b>Nihai Model Ağırlıkları:</b> <code>artifacts/champion_chemberta_cross_attention_gnn.pt</code><br/>"
        "• <b>ChemBERTa Nitelik Matrisi:</b> <code>artifacts/drug_features_chemberta.parquet</code> (3,248 Boyut)<br/>"
        "• <b>Model Mimarisi:</b> <code>src/models/chemberta_cross_attention_gnn.py</code> | <b>Eğitim:</b> <code>train_chemberta_cross_attention_model.py</code>"
    )
    files_table = Table([[Paragraph(files_text, code_style)]], colWidths=[523])
    files_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F1F5F9')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(files_table)
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Updated Architecture Report PDF created: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    generate_architecture_report_pdf()
