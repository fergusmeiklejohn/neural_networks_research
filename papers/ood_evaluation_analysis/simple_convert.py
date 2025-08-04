"""
Simple markdown to HTML converter for academic paper
"""


import markdown


def convert_to_html():
    # Read markdown
    with open("full_paper_combined.md", "r") as f:
        md_content = f.read()

    # Convert to HTML
    md = markdown.Markdown(extensions=["tables", "fenced_code", "extra"])
    body_html = md.convert(md_content)

    # Create complete HTML with styling
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>OOD Evaluation Analysis</title>
    <style>
        body {{
            font-family: 'Times New Roman', Times, serif;
            max-width: 8.5in;
            margin: 0 auto;
            padding: 1in;
            line-height: 1.6;
        }}
        h1 {{ font-size: 24pt; text-align: center; margin-bottom: 1em; }}
        h2 {{ font-size: 18pt; margin-top: 1.5em; }}
        h3 {{ font-size: 14pt; margin-top: 1em; }}
        h4 {{ font-size: 12pt; font-style: italic; }}
        p {{ text-align: justify; margin-bottom: 1em; }}
        table {{
            border-collapse: collapse;
            margin: 1em auto;
            font-size: 10pt;
        }}
        th, td {{
            border: 1px solid #666;
            padding: 0.5em;
            text-align: left;
        }}
        th {{ background-color: #f0f0f0; font-weight: bold; }}
        pre {{
            background: #f5f5f5;
            padding: 1em;
            overflow-x: auto;
            font-size: 9pt;
        }}
        code {{
            font-family: 'Courier New', monospace;
            background: #f5f5f5;
        }}
        @media print {{
            body {{ font-size: 11pt; }}
            h1, h2 {{ page-break-after: avoid; }}
            table {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
{body_html}
</body>
</html>"""

    # Write output
    with open("paper_for_review.html", "w") as f:
        f.write(html)

    print("✅ Created: paper_for_review.html")
    print("\n📄 To create PDF:")
    print("1. Open paper_for_review.html in Chrome/Safari/Firefox")
    print("2. Press Cmd+P to print")
    print("3. Select 'Save as PDF'")
    print("4. Recommended settings:")
    print("   - Margins: Default or Normal")
    print("   - Scale: 100%")
    print("   - Background graphics: OFF")
    print("5. Save as 'ood_evaluation_analysis.pdf'")


if __name__ == "__main__":
    convert_to_html()
