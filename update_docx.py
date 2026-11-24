import re
from docx import Document
from docx.shared import Pt
import os

md_path = r'C:\Users\sapur\.gemini\antigravity\brain\e17415fb-2303-4d58-8378-7785c6e839cf\problem_statement_objectives.md'
docx_path = r'e:\CU_project\AI_assistant\project_draft.docx'

# Always create a new document to overwrite the old one
doc = Document()

with open(md_path, 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
for line in lines:
    line = line.strip()
    if not line:
        continue
    
    if line.startswith('# '):
        try:
            doc.add_heading(line[2:], level=1)
        except:
            p = doc.add_paragraph()
            p.add_run(line[2:]).bold = True
    elif line.startswith('## '):
        try:
            doc.add_heading(line[3:], level=2)
        except:
            p = doc.add_paragraph()
            p.add_run(line[3:]).bold = True
    elif line.startswith('### '):
        try:
            doc.add_heading(line[4:], level=3)
        except:
            p = doc.add_paragraph()
            p.add_run(line[4:]).bold = True
    elif line.startswith('- '):
        try:
            p = doc.add_paragraph(style='List Bullet')
        except:
            p = doc.add_paragraph()
        p.add_run(line[2:])
    elif re.match(r'^\d+\.\s', line):
        try:
            p = doc.add_paragraph(style='List Number')
        except:
            p = doc.add_paragraph()
        p.add_run(line[re.search(r'^\d+\.\s', line).end():])
    elif line == '---' or line == '***':
        continue
    elif line.startswith('[Insert Figure'):
        p = doc.add_paragraph()
        run = p.add_run(line)
        run.italic = True
    elif line.startswith('*Figure'):
        p = doc.add_paragraph()
        run = p.add_run(line)
        run.italic = True
    else:
        # replace bold markdown **text** with docx bold
        p = doc.add_paragraph()
        parts = re.split(r'(\*\*.*?\*\*)', line)
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                run = p.add_run(part[2:-2])
                run.bold = True
            else:
                # also handle inline code `text`
                subparts = re.split(r'(`.*?`)', part)
                for subpart in subparts:
                    if subpart.startswith('`') and subpart.endswith('`'):
                        run = p.add_run(subpart[1:-1])
                        run.font.name = 'Courier New'
                    else:
                        p.add_run(subpart)

doc.save(docx_path)
print("Successfully updated docx file")
