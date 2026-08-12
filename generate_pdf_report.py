import os
from src.training.reproducibility import repository_root

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def build_pdf_report():
    pdf_path = str(repository_root()) + "/Polypharmacy_AI_2_Seviyeli_Model_Rehberi.pdf"
    
    pdfmetrics.registerFont(TTFont('Arial', '/System/Library/Fonts/Supplemental/Arial.ttf'))
    pdfmetrics.registerFont(TTFont('Arial-Bold', '/System/Library/Fonts/Supplemental/Arial Bold.ttf'))
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=30,
        rightMargin=30,
        topMargin=25,
        bottomMargin=25
    )
    
    styles = getSampleStyleSheet()
    
    c_primary = colors.HexColor('#0F172A')
    c_accent = colors.HexColor('#2563EB')
    c_card_bg = colors.HexColor('#F8FAFC')
    c_border = colors.HexColor('#E2E8F0')
    c_text = colors.HexColor('#334155')
    
    title_style = ParagraphStyle(
        'DocTitle',
        fontName='Arial-Bold',
        fontSize=18,
        leading=22,
        textColor=c_primary
    )
    
    section_h1 = ParagraphStyle(
        'SectionH1',
        fontName='Arial-Bold',
        fontSize=11.5,
        leading=14,
        textColor=c_accent
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        fontName='Arial',
        fontSize=8.2,
        leading=11,
        textColor=c_text
    )
    
    bold_style = ParagraphStyle(
        'BoldCustom',
        fontName='Arial-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=c_primary
    )
    
    story = []
    
    # Header Table
    header_data = [
        [
            Paragraph("<b>Çoklu İlaç Yan Etki Yapay Zekası</b><br/><font size=11 color='#2563EB'>2 Seviyeli Hiyerarşik Model Rehberi</font>", title_style),
            Paragraph("<font color='#0D9488'><b>● MODEL DURUMU: AKTİF</b></font><br/><font color='#64748B'>SIDER 4.1 + PrimeKG + ChEMBL 34</font>", ParagraphStyle('HRight', fontName='Arial', fontSize=8, leading=11, alignment=2))
        ]
    ]
    header_table = Table(header_data, colWidths=[360, 175])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceBefore=4, spaceAfter=8))
    
    # Overview Box: Neden 2 Seviyeli Mimari?
    overview_text = (
        "<b>🎯 Neden 2 Seviyeli Bir Mimari Kuruldu?</b><br/>"
        "Tıpta binlerce farklı yan etki vardır. Hekime doğrudan yüzlerce küçük yan etki listesi vermek kafa karışıklığına ve "
        "<b>'alarm yorgunluğuna'</b> yol açar. Modelimiz tıpkı bir <b>uzman hekim gibi</b> iki aşamada karar verir: "
        "<b>(1) Önce büyük resme bakar:</b> <i>'Hangi organ sistemi tehlikede?'</i> → "
        "<b>(2) Sonra detaya iner:</b> <i>'O organda tam olarak hangi klinik tanı oluşacak?'</i>"
    )
    overview_table = Table([[Paragraph(overview_text, body_style)]], colWidths=[535])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#EFF6FF')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#BFDBFE')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 8))
    
    # 2 Levels Comparison Cards
    l1_content = [
        Paragraph("<b>🏛️ SEVİYE 1: ORGAN SİSTEMİ RİSKİ</b>", section_h1),
        Paragraph("<font color='#16A34A'><b>%89.21 Doğruluk</b></font> | <font color='#2563EB'><b>%96.01 AUROC</b></font> | <font color='#7C3AED'><b>0.9216 AUPRC</b></font>", bold_style),
        Spacer(1, 3),
        Paragraph("<b>Görevi:</b> İki ilaç alındığında <b>15 temel organ sisteminin</b> hangisinde hasar riski olduğunu tespit eder.", body_style),
        Spacer(1, 3),
        Paragraph("<b>İncelenen 15 Temel Organ/Sistem:</b>", bold_style),
        Paragraph("• <b>Kalp-Damar:</b> Kriz, aritmi, tansiyon bozukluğu<br/>"
                  "• <b>Karaciğer:</b> Enzim yükü, hepatotoksisite<br/>"
                  "• <b>Mide-Bağırsak:</b> Ülser, kanama, sindirim<br/>"
                  "• <b>Böbrek:</b> Süzme bozukluğu, yetmezlik<br/>"
                  "• <b>Beyin & Sinir:</b> Felç, bilinç kaybı, nöropati<br/>"
                  "• <b>Solunum (Akciğer):</b> Nefes darlığı, astım<br/>"
                  "• <b>Kan & Lenf:</b> Anemi, pıhtılaşma bozukluğu<br/>"
                  "• <b>Deri & Cilt:</b> Alerjik ağır döküntüler<br/>"
                  "• <b>Psikiyatrik / Metabolizma / İskelet / vb.</b>", body_style),
    ]
    
    l2_content = [
        Paragraph("<b>🔬 SEVİYE 2: SPESİFİK YAN ETKİ TEŞHİSİ</b>", section_h1),
        Paragraph("<font color='#16A34A'><b>%88.43 AUROC</b></font> | <font color='#7C3AED'><b>0.5147 AUPRC</b></font> | <font color='#2563EB'><b>0.2543 P@5</b></font>", bold_style),
        Spacer(1, 3),
        Paragraph("<b>Görevi:</b> <b>Hiyerarşik Koşullu Maskeleme (Gating)</b> ile tehlikedeki organ altındaki 100 spesifik klinik tanıyı puanlar.", body_style),
        Spacer(1, 3),
        Paragraph("<b>Örnek Spesifik Klinik Teşhisler:</b>", bold_style),
        Paragraph("• <b>Aritmi & Taşikardi:</b> Kalp ritim sapması<br/>"
                  "• <b>Hepatotoksisite:</b> ALT/AST transaminaz artışı<br/>"
                  "• <b>Gastrointestinal Kanama:</b> Mide kanaması<br/>"
                  "• <b>Trombositopeni:</b> Kandaki pıhtı hücresi düşüşü<br/>"
                  "• <b>Akut Böbrek Hasarı:</b> Kreatinin yükselmesi<br/>"
                  "• <b>Bronkospazm:</b> Akciğer hava yolu tıkanması<br/>"
                  "• <b>Rabdomiyoliz:</b> Ağır kas dokusu erimesi<br/>"
                  "• <b>Stevens-Johnson:</b> Ağır cilt soyulması<br/>"
                  "• <b>Titreme (Tremor), Sarılık, vb. 100 Teşhis</b>", body_style),
    ]
    
    cards_table = Table([[l1_content, l2_content]], colWidths=[262, 263])
    cards_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), c_card_bg),
        ('BACKGROUND', (1,0), (1,0), c_card_bg),
        ('BOX', (0,0), (0,0), 1, c_border),
        ('BOX', (1,0), (1,0), 1, c_border),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(cards_table)
    story.append(Spacer(1, 8))
    
    # Clinical Scenario Simulation Box
    scenario_header = Paragraph("<b>📋 KLİNİK KARAR DESTEK SİMÜLASYONU (Doktor Ekranı)</b>", ParagraphStyle('ScHead', fontName='Arial-Bold', fontSize=9.5, textColor=c_primary))
    scenario_body = Paragraph(
        "<b>Hasta Reçetesi:</b> <font color='#2563EB'><b>İlaç A + İlaç B</b></font><br/>"
        "<b>1. Adım (Organ Riski):</b> <font color='#DC2626'><b>🔴 Karaciğer (%94 Yüksek Risk)</b></font> | "
        "<font color='#DC2626'><b>🔴 Kalp-Damar (%88 Yüksek Risk)</b></font> | "
        "<font color='#16A34A'><b>🟢 Böbrekler (%12 Güvenli)</b></font><br/>"
        "<b>2. Adım (Gated Spesifik Teşhis):</b> "
        "<i>Karaciğer Altında:</i> <b>ALT/AST Enzim Artışı (%92)</b>, <b>Sarılık (%85)</b> | "
        "<i>Kalp Altında:</i> <b>Taşikardi (%89)</b>, <b>Hipotansiyon (%81)</b>",
        body_style
    )
    scenario_table = Table([[scenario_header], [scenario_body]], colWidths=[535])
    scenario_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FEF2F2')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#FECACA')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(scenario_table)
    story.append(Spacer(1, 8))
    
    # Footer Banner
    footer_text = (
        "<b>💡 MODELİN 3 BÜYÜK KLİNİK DEĞERİ:</b> "
        "<b>1. Hayat Kurtarır:</b> Toksik kombinasyonları saniyeler içinde yakalar. | "
        "<b>2. Alarm Yorgunluğunu Önler:</b> Gating ile doğrudan riskli organa odaklar. | "
        "<b>3. Bilimsel Güç:</b> 2.42M Molekül (ChEMBL 34), 9.61M Bağ (PrimeKG), SIDER 4.1 Ön Bilgileri."
    )
    footer_table = Table([[Paragraph(footer_text, ParagraphStyle('Foot', fontName='Arial', fontSize=7.5, leading=10, textColor=colors.HexColor('#475569')))]], colWidths=[535])
    footer_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F1F5F9')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(footer_table)
    
    doc.build(story)
    print(f"Updated PDF generated: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    build_pdf_report()
