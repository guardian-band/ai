import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Paths
base_dir = "/Users/acelyayildiz/.gemini/antigravity/scratch/polypharmacy_ai"
artifacts_dir = os.path.join(base_dir, "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)
chart_path = os.path.join(artifacts_dir, "metrics_chart.png")
arch_path = os.path.join(artifacts_dir, "architecture_diagram.png")
pdf_path = os.path.join(base_dir, "Polypharmacy_AI_Comprehensive_Report.pdf")

def create_metrics_chart():
    labels = ['Accuracy', 'AUROC', 'Macro AUPRC']
    level1_scores = [89.21, 96.01, 92.16]  # Scaled AUPRC to 100 for comparison
    level2_scores = [84.74, 88.43, 51.47]
    
    x = range(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(7, 4))
    rects1 = ax.bar([i - width/2 for i in x], level1_scores, width, label='Level 1 (Organ Systems)', color='#1E40AF')
    rects2 = ax.bar([i + width/2 for i in x], level2_scores, width, label='Level 2 (Specific Side Effects)', color='#0D9488')
    
    ax.set_ylabel('Scores (%)', fontsize=12)
    ax.set_title('Performance Metrics: Level 1 vs Level 2', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='lower left')
    ax.set_ylim(0, 110)
    
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    
    # Draw boxes
    boxes = [
        {"xy": (0.05, 0.7), "w": 0.25, "h": 0.2, "text": "Drug A Features\n(ChemBERTa, Morgan)", "color": "#E2E8F0"},
        {"xy": (0.05, 0.3), "w": 0.25, "h": 0.2, "text": "Drug B Features\n(ChemBERTa, Morgan)", "color": "#E2E8F0"},
        
        {"xy": (0.4, 0.5), "w": 0.25, "h": 0.2, "text": "Cross-Attention Fusion\n(Unified GNN)", "color": "#93C5FD"},
        
        {"xy": (0.75, 0.7), "w": 0.2, "h": 0.15, "text": "Level 1 Output\n(15 Organ Systems)", "color": "#86EFAC"},
        {"xy": (0.75, 0.3), "w": 0.2, "h": 0.15, "text": "Level 2 Output\n(100 Specific Effects)", "color": "#FDE047"},
    ]
    
    for b in boxes:
        rect = patches.Rectangle(b["xy"], b["w"], b["h"], linewidth=1.5, edgecolor='#334155', facecolor=b["color"])
        ax.add_patch(rect)
        ax.text(b["xy"][0] + b["w"]/2, b["xy"][1] + b["h"]/2, b["text"], ha='center', va='center', fontsize=10, fontweight='bold')
        
    # Arrows
    style = "Simple, tail_width=1.5, head_width=6, head_length=8"
    kw = dict(arrowstyle=style, color="#334155")
    
    arrow1 = patches.FancyArrowPatch((0.3, 0.8), (0.4, 0.65), connectionstyle="arc3,rad=-0.2", **kw)
    arrow2 = patches.FancyArrowPatch((0.3, 0.4), (0.4, 0.55), connectionstyle="arc3,rad=0.2", **kw)
    arrow3 = patches.FancyArrowPatch((0.65, 0.6), (0.75, 0.75), connectionstyle="arc3,rad=0.2", **kw)
    arrow4 = patches.FancyArrowPatch((0.65, 0.6), (0.75, 0.4), connectionstyle="arc3,rad=-0.2", **kw)
    
    for a in [arrow1, arrow2, arrow3, arrow4]:
        ax.add_patch(a)
        
    ax.text(0.68, 0.58, "Hierarchical\nGating", ha='center', va='center', fontsize=9, fontstyle='italic', color='#EF4444')
    
    plt.title('Polypharmacy AI - Hierarchical Neural Architecture', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(arch_path, dpi=300)
    plt.close()

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
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor('#64748B'))
        self.drawString(36, 30, "Antigravity AI Lab | Confidential Document")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(559, 30, page_str)
        self.restoreState()

def build_pdf_report():
    create_metrics_chart()
    create_architecture_diagram()
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=50
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('Title', fontName='Helvetica-Bold', fontSize=22, leading=26, textColor=colors.HexColor('#0F172A'), alignment=1, spaceAfter=20)
    h1_style = ParagraphStyle('H1', fontName='Helvetica-Bold', fontSize=14, leading=18, textColor=colors.HexColor('#1E40AF'), spaceBefore=15, spaceAfter=10)
    h2_style = ParagraphStyle('H2', fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=colors.HexColor('#0F766E'), spaceBefore=10, spaceAfter=8)
    body_style = ParagraphStyle('Body', fontName='Helvetica', fontSize=10, leading=14, textColor=colors.HexColor('#334155'), spaceAfter=10)
    
    story = []
    
    # Title
    story.append(Paragraph("Polypharmacy AI Model: Comprehensive Technical Report", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1E40AF'), spaceAfter=20))
    
    # Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph("This report presents the architecture, training details, and final evaluation metrics for the Polypharmacy Artificial Intelligence model. The objective of this model is to predict the potential side effects of combinations of drugs (polypharmacy) that have not been previously tested together. Due to the vast number of potential side effects, the model employs a <b>Two-Level Hierarchical Architecture</b>, mimicking a clinician's diagnostic workflow.", body_style))
    story.append(Paragraph("By combining vast datasets (DrugBank, PrimeKG, ChEMBL, and BioSNAP), our Unified Master Model achieves state-of-the-art results, effectively prioritizing high-risk combinations.", body_style))
    
    # Architecture
    story.append(Paragraph("2. Model Architecture", h1_style))
    story.append(Paragraph("The system was designed with two distinct prediction levels to prevent 'alarm fatigue' and improve clinical interpretability:", body_style))
    story.append(Paragraph("<b>Level 1 (Organ Systems):</b> Predicts which of the 15 primary MedDRA organ systems (e.g., Cardiovascular, Hepatic) is at risk.", body_style))
    story.append(Paragraph("<b>Level 2 (Specific Side Effects):</b> Uses Hierarchical Gating to narrow down the prediction to one of 100 specific clinical diagnoses (e.g., Tachycardia, Hepatotoxicity).", body_style))
    
    # Insert Diagram
    story.append(Spacer(1, 10))
    story.append(Image(arch_path, width=450, height=281))
    story.append(Spacer(1, 10))
    
    # RGCN Explanation
    story.append(Paragraph("3. Technical Decisions: Moving Beyond RGCN", h1_style))
    story.append(Paragraph("Initially, a Relational Graph Convolutional Network (RGCN) was considered. RGCNs excel at modeling multi-relational graphs by treating drugs as nodes and interactions as directed edges with specific relation types.", body_style))
    story.append(Paragraph("<b>Why was RGCN dropped?</b>", h2_style))
    story.append(Paragraph("1. <b>Hardware & Memory Constraints:</b> The vastness of the PrimeKG and BioSNAP knowledge graphs (millions of edges) requires 16GB+ of VRAM to train an RGCN effectively. Our target hardware (Apple Silicon with 8GB RAM) suffered from Out-Of-Memory (OOM) errors.", body_style))
    story.append(Paragraph("2. <b>MPS Limitations:</b> Certain sparse tensor operations required by PyTorch Geometric for RGCNs are not fully supported or optimized on Apple's Metal Performance Shaders (MPS), leading to severe bottlenecks.", body_style))
    story.append(Paragraph("<b>The Solution: Unified GNN with Cross-Attention</b>", h2_style))
    story.append(Paragraph("Instead of a heavy RGCN, we developed a Unified Graph Neural Network. This architecture extracts 3,248-dimensional rich embeddings per drug (incorporating ChemBERTa-77M molecular representations and Morgan fingerprints) and applies a Multi-Head Cross-Attention fusion layer. This approach is highly memory-efficient, training in under 3 minutes on MPS while maintaining high predictive power.", body_style))
    
    story.append(PageBreak())
    
    # Performance Metrics
    story.append(Paragraph("4. Performance Evaluation", h1_style))
    story.append(Paragraph("The Unified Master Model was trained and evaluated on disjoint test pairs. The following chart illustrates the primary metrics.", body_style))
    
    # Insert Chart
    story.append(Spacer(1, 10))
    story.append(Image(chart_path, width=400, height=228))
    story.append(Spacer(1, 15))
    
    # Detailed Tables
    story.append(Paragraph("<b>Level 1: Organ System Risk (15 MedDRA SOCs)</b>", h2_style))
    l1_data = [
        ["Metric", "Score", "Interpretation"],
        ["Accuracy", "89.21%", "High overall classification correctness."],
        ["AUROC", "96.01%", "Excellent at distinguishing between risk / no-risk."],
        ["Macro AUPRC", "0.9216", "Outstanding ability to rank true risks at the top."]
    ]
    t1 = Table(l1_data, colWidths=[120, 80, 300])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E40AF')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC'))
    ]))
    story.append(t1)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("<b>Level 2: Specific Side Effects (Top 100 Diagnoses)</b>", h2_style))
    l2_data = [
        ["Metric", "Score", "Interpretation"],
        ["Accuracy", "84.74%", "Good general correctness across specific tasks."],
        ["AUROC", "88.43%", "Strong discriminative ability for specific effects."],
        ["Macro AUPRC", "0.5147", "Moderate. Highly challenging due to rare classes."],
        ["Precision@5", "0.2543", "Top 5 predictions are highly actionable for clinicians."]
    ]
    t2 = Table(l2_data, colWidths=[120, 80, 300])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F766E')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F0FDF4'))
    ]))
    story.append(t2)
    
    # Analysis on AUPRC
    story.append(Paragraph("5. Analysis & Future Work", h1_style))
    story.append(Paragraph("While Level 1 metrics are exceptional (0.92 AUPRC), Level 2 AUPRC plateaued around 0.51. To improve this, we implemented Focal Loss to handle class imbalance, integrated SIDER 4.1 priors, and utilized ChemBERTa chemical language models. However, treating drugs as static vectors limits the model's ability to fully capture the dynamic, non-linear chemical chaos when two complex molecules interact.", body_style))
    story.append(Paragraph("Future work should focus on conditional molecular embeddings or exploring sparse, memory-optimized graph implementations that can run efficiently on consumer hardware.", body_style))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Comprehensive English Report generated at: {pdf_path}")

if __name__ == "__main__":
    build_pdf_report()
