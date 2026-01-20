"""
Générateur de rapport PDF professionnel
Style: sobre, élégant, type Palantir
"""

import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
import matplotlib
matplotlib.use('Agg')

class DataQualityReport:
    """
    Générateur de rapport de qualité des données
    Style professionnel, sobre et élégant
    """
    
    # Palette de couleurs professionnelle (inspirée Palantir)
    COLOR_PRIMARY = colors.HexColor('#1a1a2e')      # Bleu très foncé
    COLOR_SECONDARY = colors.HexColor('#16213e')    # Bleu foncé
    COLOR_ACCENT = colors.HexColor('#0f4c75')       # Bleu moyen
    COLOR_HIGHLIGHT = colors.HexColor('#3282b8')    # Bleu clair
    COLOR_TEXT = colors.HexColor('#2d2d2d')         # Gris très foncé
    COLOR_TEXT_LIGHT = colors.HexColor('#666666')   # Gris moyen
    COLOR_BACKGROUND = colors.HexColor('#f8f9fa')   # Gris très clair
    COLOR_WARNING = colors.HexColor('#d63031')      # Rouge
    COLOR_SUCCESS = colors.HexColor('#00b894')      # Vert
    
    def __init__(self, filename="data_quality_report.pdf"):
        self.filename = filename
        self.doc = SimpleDocTemplate(
            filename,
            pagesize=A4,
            rightMargin=50,
            leftMargin=50,
            topMargin=80,
            bottomMargin=50
        )
        self.story = []
        self.styles = getSampleStyleSheet()
        self._setup_styles()
    
    def _setup_styles(self):
        """Configure les styles personnalisés."""
        
        # Titre principal
        if 'CustomTitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=28,
                textColor=self.COLOR_PRIMARY,
                spaceAfter=30,
                alignment=TA_LEFT,
                fontName='Helvetica-Bold'
            ))
        
        # Sous-titre
        if 'CustomSubtitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomSubtitle',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=self.COLOR_TEXT_LIGHT,
                spaceAfter=20,
                alignment=TA_LEFT
            ))
        
        # En-tête de section
        if 'SectionHeader' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading2'],
                fontSize=16,
                textColor=self.COLOR_ACCENT,
                spaceAfter=12,
                spaceBefore=20,
                fontName='Helvetica-Bold',
                borderWidth=0,
                borderColor=self.COLOR_ACCENT,
                borderPadding=0,
                leftIndent=0
            ))
        
        # Corps de texte
        if 'CustomBody' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomBody',
                parent=self.styles['Normal'],
                fontSize=10,
                textColor=self.COLOR_TEXT,
                spaceAfter=6,
                leading=14
            ))
        
        # Métrique
        if 'Metric' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='Metric',
                parent=self.styles['Normal'],
                fontSize=10,
                textColor=self.COLOR_TEXT,
                spaceAfter=4,
                leftIndent=20
            ))
    
    def _add_header(self, canvas, doc):
        """Ajoute un en-tête sur chaque page."""
        canvas.saveState()
        
        # Ligne de séparation
        canvas.setStrokeColor(self.COLOR_ACCENT)
        canvas.setLineWidth(2)
        canvas.line(50, A4[1] - 50, A4[0] - 50, A4[1] - 50)
        
        # Numéro de page
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(self.COLOR_TEXT_LIGHT)
        canvas.drawRightString(
            A4[0] - 50,
            A4[1] - 65,
            f"Page {doc.page}"
        )
        
        canvas.restoreState()
    
    def add_cover_page(self, dataset_name, total_rows, total_cols):
        """Ajoute une page de couverture."""
        
        # Titre principal
        title = Paragraph(
            "DATA QUALITY REPORT",
            self.styles['CustomTitle']
        )
        self.story.append(title)
        self.story.append(Spacer(1, 0.3 * inch))
        
        # Informations du dataset
        subtitle = Paragraph(
            f"Dataset: <b>{dataset_name}</b>",
            self.styles['CustomSubtitle']
        )
        self.story.append(subtitle)
        
        info = Paragraph(
            f"{total_rows:,} rows × {total_cols} columns",
            self.styles['CustomSubtitle']
        )
        self.story.append(info)
        
        # Date du rapport
        date_text = Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles['CustomSubtitle']
        )
        self.story.append(date_text)
        
        self.story.append(Spacer(1, 1.5 * inch))
        
        # Résumé visuel
        self._add_summary_box(total_rows, total_cols)
        
        self.story.append(PageBreak())
    
    def _add_summary_box(self, rows, cols):
        """Ajoute une box de résumé."""
        data = [
            ['DATASET OVERVIEW', ''],
            ['Total Records', f'{rows:,}'],
            ['Total Columns', str(cols)],
            ['Report Type', 'Quality Assessment']
        ]
        
        table = Table(data, colWidths=[3 * inch, 2 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.COLOR_PRIMARY),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, self.COLOR_TEXT_LIGHT),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white)
        ]))
        
        self.story.append(table)
    
    def add_section(self, title, content_items):
        """Ajoute une section avec titre et contenu."""
        
        # Titre de section
        section_title = Paragraph(title, self.styles['SectionHeader'])
        self.story.append(section_title)
        
        # Contenu
        for item in content_items:
            if isinstance(item, str):
                p = Paragraph(item, self.styles['CustomBody'])
                self.story.append(p)
            else:
                self.story.append(item)
        
        self.story.append(Spacer(1, 0.2 * inch))
    
    def add_anomaly_table(self, title, data, headers):
        """Ajoute un tableau d'anomalies."""
        
        section_title = Paragraph(title, self.styles['SectionHeader'])
        self.story.append(section_title)
        
        if not data:
            no_data = Paragraph(
                "No anomalies detected.",
                self.styles['CustomBody']
            )
            self.story.append(no_data)
            self.story.append(Spacer(1, 0.2 * inch))
            return
        
        # Créer le tableau
        table_data = [headers] + data
        
        col_widths = [1.2 * inch] + [1.8 * inch] * (len(headers) - 1)
        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        
        table.setStyle(TableStyle([
            # En-tête
            ('BACKGROUND', (0, 0), (-1, 0), self.COLOR_ACCENT),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            
            # Corps
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, self.COLOR_TEXT_LIGHT),
            
            # Alternance de couleurs
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.COLOR_BACKGROUND])
        ]))
        
        self.story.append(table)
        self.story.append(Spacer(1, 0.3 * inch))
    
    def add_chart(self, chart_path, width=5*inch, height=3*inch):
        """Ajoute un graphique."""
        from reportlab.platypus import Image as RLImage
        img = RLImage(chart_path, width=width, height=height)
        self.story.append(img)
        self.story.append(Spacer(1, 0.2 * inch))
    
    def generate(self):
        """Génère le PDF."""
        self.doc.build(
            self.story,
            onFirstPage=self._add_header,
            onLaterPages=self._add_header
        )
        print(f"Report generated: {self.filename}")


def create_quality_report(df, output_file="data_quality_report.pdf"):
    """
    Crée un rapport de qualité professionnelpour un DataFrame.
    """
    from Cleaner import Cleaner
    
    # Générer le rapport d'anomalies
    cleaner = Cleaner()
    report = cleaner.generate_report(df)
    
    # Créer le PDF
    pdf = DataQualityReport(output_file)
    
    # Page de couverture
    pdf.add_cover_page(
        dataset_name="company_data.csv",
        total_rows=report['total_rows'],
        total_cols=report['total_cols']
    )
    
    # Section 1: Executive Summary
    summary_content = []
    total_issues = (
        len(report.get('missing_values', {})) +
        len(report.get('outliers', {})) +
        len(report.get('sequence_breaks', {})) +
        sum(len(v) for v in report.get('domain_anomalies', {}).values()) +
        len(report.get('duplicates', {}).get('complete_duplicates', [])) +
        sum(len(v) for v in report.get('invalid_formats', {}).values()) +
        sum(len(v) for v in report.get('impossible_values', {}).values()) +
        len(report.get('business_inconsistencies', {}).get('inconsistencies', []))
    )
    
    summary_content.append(
        f"This report provides a comprehensive analysis of data quality issues detected in the dataset. "
        f"A total of <b>{total_issues}</b> anomalies were identified across multiple categories."
    )
    summary_content.append("")
    summary_content.append("<b>Key Findings:</b>")
    summary_content.append(f"• Missing values: {sum(len(v) for v in report['missing_values'].values())} occurrences")
    summary_content.append(f"• Statistical outliers: {sum(len(v) for v in report['outliers'].values())} values")
    summary_content.append(f"• Sequence breaks: {len(report['sequence_breaks'])} columns")
    summary_content.append(f"• Domain anomalies: {sum(len(v) for v in report.get('domain_anomalies', {}).values())} emails")
    summary_content.append(f"• Complete duplicates: {len(report.get('duplicates', {}).get('complete_duplicates', []))} rows")
    summary_content.append(f"• Invalid formats: {sum(len(v) for v in report.get('invalid_formats', {}).values())} values")
    summary_content.append(f"• Impossible values: {sum(len(v) for v in report.get('impossible_values', {}).values())} values")
    summary_content.append(f"• Business inconsistencies: {len(report.get('business_inconsistencies', {}).get('inconsistencies', []))} issues")
    
    pdf.add_section("EXECUTIVE SUMMARY", summary_content)
    
    # Section 2: Missing Values
    if report['missing_values']:
        missing_data = []
        for col, rows in report['missing_values'].items():
            missing_data.append([
                col,
                str(len(rows)),
                f"{len(rows)/report['total_rows']*100:.1f}%",
                str(rows[:3])[1:-1] + ('...' if len(rows) > 3 else '')
            ])
        
        pdf.add_anomaly_table(
            "MISSING VALUES ANALYSIS",
            missing_data,
            ['Column', 'Count', '% of Total', 'Sample Rows']
        )
    
    # Section 3: Outliers
    if report['outliers']:
        for col, outliers in report['outliers'].items():
            outlier_data = []
            for item in outliers[:10]:  # Limiter à 10
                outlier_data.append([
                    str(item['ligne']),
                    f"{item['valeur']:.2f}",
                    f"{item['min_attendu']:.2f}",
                    f"{item['max_attendu']:.2f}"
                ])
            
            pdf.add_anomaly_table(
                f"OUTLIERS: {col.upper()}",
                outlier_data,
                ['Row', 'Value', 'Min Expected', 'Max Expected']
            )
    
    # Section 4: Sequence Breaks
    if report['sequence_breaks']:
        seq_data = []
        for col, brk in report['sequence_breaks'].items():
            seq_data.append([
                col,
                str(brk['ligne']),
                brk['regle'],
                brk['attendu'],
                brk['observe']
            ])
        
        pdf.add_anomaly_table(
            "SEQUENCE ANOMALIES",
            seq_data,
            ['Column', 'Row', 'Rule', 'Expected', 'Observed']
        )
    
    # Section 5: Domain Anomalies
    if report.get('domain_anomalies'):
        for col, anomalies in report['domain_anomalies'].items():
            domain_data = []
            for item in anomalies[:10]:
                domain_data.append([
                    str(item['ligne']),
                    item['email'][:30] + '...' if len(item['email']) > 30 else item['email'],
                    item['domaine'],
                    item['domaine_attendu']
                ])
            
            pdf.add_anomaly_table(
                f"DOMAIN ANOMALIES: {col.upper()}",
                domain_data,
                ['Row', 'Email', 'Domain', 'Expected']
            )
    
    # Section 6: Duplicates
    if report['duplicates']['complete_duplicates']:
        dup_data = []
        dup_rows = report['duplicates']['complete_duplicates'][:10]
        for row in dup_rows:
            dup_data.append([str(row)])
        
        pdf.add_anomaly_table(
            "COMPLETE DUPLICATES",
            dup_data,
            ['Duplicate Rows']
        )
    
    if report['duplicates']['partial_duplicates']:
        for col, dup_info in report['duplicates']['partial_duplicates'].items():
            dup_data = []
            rows = dup_info['rows'][:10]
            for row in rows:
                dup_data.append([str(row), str(dup_info['values'].get(report['duplicates']))])
            
            pdf.add_anomaly_table(
                f"PARTIAL DUPLICATES: {col.upper()}",
                dup_data,
                ['Row', 'Duplicate Value']
            )
    
    # Section 7: Invalid Formats
    if report.get('invalid_formats'):
        for col, issues in report['invalid_formats'].items():
            format_data = []
            for item in issues[:10]:
                format_data.append([
                    str(item['ligne']),
                    item['valeur'][:40] + '...' if len(item['valeur']) > 40 else item['valeur']
                ])
            
            pdf.add_anomaly_table(
                f"INVALID FORMATS: {col.upper()}",
                format_data,
                ['Row', 'Invalid Value']
            )
    
    # Section 8: Impossible Values
    if report.get('impossible_values'):
        for col, issues in report['impossible_values'].items():
            imp_data = []
            for item in issues[:10]:
                imp_data.append([
                    str(item['ligne']),
                    f"{item['valeur']:.2f}" if isinstance(item['valeur'], float) else str(item['valeur'])
                ])
            
            pdf.add_anomaly_table(
                f"IMPOSSIBLE VALUES: {col.upper()}",
                imp_data,
                ['Row', 'Value']
            )
    
    # Section 9: Advanced Statistics
    if report.get('advanced_statistics') and 'error' not in report['advanced_statistics']:
        stats_text = []
        stats_text.append("<b>Distribution Analysis:</b>")
        stats_text.append("")
        
        for col, stats in report['advanced_statistics'].items():
            stats_text.append(f"<b>{col}:</b>")
            stats_text.append(f"  • Skewness: {stats['skewness']:.3f} ({stats['distribution_type']})")
            stats_text.append(f"  • Kurtosis: {stats['kurtosis']:.3f}")
            stats_text.append(f"  • Coefficient of Variation: {stats['coefficient_variation']:.3f}")
            stats_text.append("")
        
        pdf.add_section("STATISTICAL ANALYSIS", stats_text)
    
    # Section 10: Business Inconsistencies
    if report.get('business_inconsistencies', {}).get('inconsistencies'):
        incons_data = []
        for item in report['business_inconsistencies']['inconsistencies'][:10]:
            incons_data.append([
                str(item['ligne']),
                item['type'],
                item['description'][:50] + '...' if len(item['description']) > 50 else item['description']
            ])
        
        pdf.add_anomaly_table(
            "BUSINESS INCONSISTENCIES",
            incons_data,
            ['Row', 'Type', 'Description']
        )
    
    # Section 11: Recommendations
    recommendations = [
        "<b>Immediate Actions:</b>",
        "• Review and correct sequence breaks in identifier columns",
        "• Investigate outlier values for data entry errors",
        "• Address missing values based on business rules",
        "• Remove duplicate records and fix format issues",
        "",
        "<b>Data Quality Improvements:</b>",
        "• Implement validation rules at data entry",
        "• Set up automated quality checks",
        "• Establish data governance policies",
        "• Add business rule validation for salaries and ages"
    ]
    
    pdf.add_section("RECOMMENDATIONS", recommendations)
    
    # Générer le PDF
    pdf.generate()
    
    return output_file


if __name__ == "__main__":
    # Test
    df = pd.read_csv("company_data.csv")
    output = create_quality_report(df, "data_quality_report.pdf")
    print(f"\nProfessional report generated: {output}")
