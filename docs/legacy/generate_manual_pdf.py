#!/usr/bin/env python3
"""
Generate MANUAL.pdf from MANUAL.md with embedded images
Uses markdown2 for parsing and reportlab for PDF generation
"""

import os
import re
from pathlib import Path

# Check if we can use pandoc (best option) or fall back to reportlab
try:
    import subprocess
    result = subprocess.run(['pandoc', '--version'], capture_output=True)
    HAS_PANDOC = result.returncode == 0
except:
    HAS_PANDOC = False

if HAS_PANDOC:
    print("Using Pandoc for high-quality PDF generation...")
    
    # Create enhanced markdown with better formatting
    with open('MANUAL.md', 'r') as f:
        content = f.read()
    
    # Add metadata for pandoc
    enhanced_md = """---
title: "DIGITON MODEM - Technical Manual & User Guide"
author: "Digiton Project Team"
date: "November 28, 2025"
geometry: margin=1in
fontsize: 11pt
documentclass: article
toc: true
toc-depth: 3
colorlinks: true
linkcolor: blue
urlcolor: blue
---

""" + content
    
    # Write enhanced markdown
    with open('MANUAL_enhanced.md', 'w') as f:
        f.write(enhanced_md)
    
    # Generate PDF with pandoc
    cmd = [
        'pandoc',
        'MANUAL_enhanced.md',
        '-o', 'MANUAL.pdf',
        '--pdf-engine=pdflatex',
        '-V', 'geometry:margin=1in',
        '-V', 'fontsize=11pt',
        '--toc',
        '--number-sections',
        '--highlight-style=tango'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ PDF generated successfully: MANUAL.pdf")
            os.remove('MANUAL_enhanced.md')
        else:
            print(f"Pandoc error: {result.stderr}")
            print("\nFalling back to reportlab method...")
            HAS_PANDOC = False
    except Exception as e:
        print(f"Error running pandoc: {e}")
        print("\nFalling back to reportlab method...")
        HAS_PANDOC = False

if not HAS_PANDOC:
    print("Pandoc not available. Using reportlab for PDF generation...")
    
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        from reportlab.pdfgen import canvas
        
        print("Generating PDF with reportlab...")
        
        # Create PDF
        doc = SimpleDocTemplate("MANUAL.pdf", pagesize=letter,
                               topMargin=0.75*inch, bottomMargin=0.75*inch,
                               leftMargin=0.75*inch, rightMargin=0.75*inch)
        
        # Container for PDF elements
        story = []
        
        # Define styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading1_style = ParagraphStyle(
            'CustomHeading1',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#2563eb'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        )
        
        heading2_style = ParagraphStyle(
            'CustomHeading2',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=10,
            spaceBefore=16,
            fontName='Helvetica-Bold'
        )
        
        heading3_style = ParagraphStyle(
            'CustomHeading3',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#1e3a8a'),
            spaceAfter=8,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
            leading=14
        )
        
        code_style = ParagraphStyle(
            'CustomCode',
            parent=styles['Code'],
            fontSize=9,
            fontName='Courier',
            textColor=colors.HexColor('#1f2937'),
            backColor=colors.HexColor('#f3f4f6'),
            leftIndent=20,
            rightIndent=20,
            spaceAfter=10
        )
        
        # Read markdown
        with open('MANUAL.md', 'r') as f:
            lines = f.readlines()
        
        # Parse markdown and build PDF
        in_code_block = False
        code_buffer = []
        
        for line in lines:
            line = line.rstrip()
            
            # Skip YAML-style metadata
            if line.startswith('---'):
                continue
            
            # Code blocks
            if line.startswith('```'):
                if in_code_block:
                    # End of code block
                    code_text = '\n'.join(code_buffer)
                    story.append(Paragraph(f'<font face="Courier" size="8">{code_text}</font>', code_style))
                    code_buffer = []
                    in_code_block = False
                else:
                    # Start of code block
                    in_code_block = True
                continue
            
            if in_code_block:
                code_buffer.append(line)
                continue
            
            # Images
            img_match = re.match(r'!\[([^\]]*)\]\(([^\)]+)\)', line)
            if img_match:
                caption = img_match.group(1)
                img_path = img_match.group(2)
                
                if os.path.exists(img_path):
                    try:
                        # Add image
                        img = Image(img_path, width=6*inch, height=4*inch)
                        story.append(img)
                        
                        # Add caption
                        if caption:
                            cap = Paragraph(f'<i>{caption}</i>', body_style)
                            story.append(cap)
                        
                        story.append(Spacer(1, 0.2*inch))
                    except Exception as e:
                        print(f"Warning: Could not add image {img_path}: {e}")
                        story.append(Paragraph(f'[Image: {caption}]', body_style))
                continue
            
            # Title (# )
            if line.startswith('# '):
                text = line[2:].strip()
                story.append(Paragraph(text, title_style))
                story.append(Spacer(1, 0.2*inch))
                continue
            
            # Heading 1 (## )
            if line.startswith('## '):
                text = line[3:].strip()
                story.append(Paragraph(text, heading1_style))
                continue
            
            # Heading 2 (### )
            if line.startswith('### '):
                text = line[4:].strip()
                story.append(Paragraph(text, heading2_style))
                continue
            
            # Heading 3 (#### )
            if line.startswith('#### '):
                text = line[5:].strip()
                story.append(Paragraph(text, heading3_style))
                continue
            
            # Horizontal rule
            if line.strip() in ['---', '***', '___']:
                story.append(Spacer(1, 0.1*inch))
                continue
            
            # Tables (simple detection)
            if '|' in line and line.strip().startswith('|'):
                # Skip for now - complex to parse
                story.append(Paragraph(line, code_style))
                continue
            
            # Bold/Italic (basic handling)
            text = line
            text = re.sub(r'\*\*([^\*]+)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'\*([^\*]+)\*', r'<i>\1</i>', text)
            text = re.sub(r'`([^`]+)`', r'<font face="Courier">\1</font>', text)
            
            # Bullet points
            if line.strip().startswith('- ') or line.strip().startswith('* '):
                text = '• ' + line.strip()[2:]
                story.append(Paragraph(text, body_style))
                continue
            
            # Numbered lists
            if re.match(r'^\d+\.\s', line.strip()):
                story.append(Paragraph(line.strip(), body_style))
                continue
            
            # Empty lines
            if not line.strip():
                story.append(Spacer(1, 0.1*inch))
                continue
            
            # Regular paragraphs
            if line.strip():
                story.append(Paragraph(text, body_style))
        
        # Build PDF
        doc.build(story)
        print("✓ PDF generated successfully: MANUAL.pdf")
        
    except ImportError:
        print("\nERROR: Neither pandoc nor reportlab is available.")
        print("\nTo generate PDF, install one of:")
        print("  Option 1 (recommended): sudo apt-get install pandoc texlive-latex-base texlive-latex-extra")
        print("  Option 2: pip install reportlab")
        exit(1)
    except Exception as e:
        print(f"\nError generating PDF: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

print("\n✓ MANUAL.pdf created successfully!")
print(f"  Location: {os.path.abspath('MANUAL.pdf')}")
print(f"  Size: {os.path.getsize('MANUAL.pdf') / 1024:.1f} KB")
