"""
Convert markdown paper to HTML with academic styling
"""

import markdown
from markdown.extensions import tables, fenced_code, codehilite, toc, attr_list, md_in_html
import re

def process_markdown(md_content):
    """Process markdown with academic paper styling"""
    
    # Configure markdown extensions
    md = markdown.Markdown(extensions=[
        'tables',
        'fenced_code',
        'codehilite',
        'toc',
        'attr_list',
        'md_in_html',
        'extra'
    ])
    
    # Convert markdown to HTML
    html_content = md.convert(md_content)
    
    # Create full HTML document with academic styling
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>An Analysis of Out-of-Distribution Evaluation in Physics-Informed Neural Networks</title>
    <style>
        /* Academic paper styling */
        body {{
            font-family: 'Times New Roman', Times, serif;
            line-height: 1.6;
            color: #333;
            max-width: 8.5in;
            margin: 0 auto;
            padding: 1in;
            background-color: white;
        }}
        
        /* Title and headers */
        h1 {
            font-size: 24pt;
            text-align: center;
            margin-bottom: 1em;
            font-weight: bold;
        }
        
        h2 {
            font-size: 18pt;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            font-weight: bold;
        }
        
        h3 {
            font-size: 14pt;
            margin-top: 1em;
            margin-bottom: 0.5em;
            font-weight: bold;
        }
        
        h4 {
            font-size: 12pt;
            margin-top: 0.8em;
            margin-bottom: 0.4em;
            font-weight: bold;
            font-style: italic;
        }
        
        /* Abstract */
        h2:first-of-type + p {
            font-style: italic;
            margin: 1em 2em;
            text-align: justify;
        }
        
        /* Body text */
        p {
            text-align: justify;
            margin-bottom: 1em;
            text-indent: 0.5in;
        }
        
        /* First paragraph after heading - no indent */
        h1 + p, h2 + p, h3 + p, h4 + p {
            text-indent: 0;
        }
        
        /* Lists */
        ul, ol {
            margin-left: 0.5in;
            margin-bottom: 1em;
        }
        
        li {
            margin-bottom: 0.3em;
        }
        
        /* Tables */
        table {
            border-collapse: collapse;
            margin: 1em auto;
            font-size: 11pt;
        }
        
        th, td {
            border: 1px solid #666;
            padding: 0.5em;
            text-align: left;
        }
        
        th {
            background-color: #f0f0f0;
            font-weight: bold;
        }
        
        /* Code blocks */
        pre {
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            padding: 1em;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 10pt;
            margin: 1em 0;
        }
        
        code {
            font-family: 'Courier New', monospace;
            font-size: 10pt;
            background-color: #f5f5f5;
            padding: 0.1em 0.3em;
        }
        
        /* Blockquotes */
        blockquote {
            margin: 1em 2em;
            padding-left: 1em;
            border-left: 3px solid #ccc;
            font-style: italic;
        }
        
        /* Page breaks for printing */
        @media print {
            h1, h2 {
                page-break-after: avoid;
            }
            
            table, figure {
                page-break-inside: avoid;
            }
            
            body {
                font-size: 11pt;
            }
        }
        
        /* Figure captions */
        p strong:first-child {
            display: block;
            text-align: center;
            margin-top: 1em;
            margin-bottom: 0.5em;
        }
        
        /* Keywords */
        p:nth-of-type(5) {
            font-style: italic;
            text-indent: 0;
            margin-top: 1em;
        }
        
        /* Section numbers */
        h2::before {
            content: counter(section) ". ";
            counter-increment: section;
        }
        
        h3::before {
            content: counter(section) "." counter(subsection) " ";
            counter-increment: subsection;
        }
        
        h2 {
            counter-reset: subsection;
        }
        
        body {
            counter-reset: section;
        }
        
        /* Don't number Abstract */
        h2:first-of-type::before {
            content: "";
            counter-increment: none;
        }
    </style>
    <script>
        // Add figure references
        window.onload = function() {
            // Number tables
            var tables = document.querySelectorAll('table');
            tables.forEach(function(table, index) {
                var caption = table.previousElementSibling;
                if (caption && caption.tagName === 'P' && caption.textContent.includes('Table')) {
                    caption.style.textAlign = 'center';
                    caption.style.fontWeight = 'bold';
                    caption.style.marginBottom = '0.5em';
                }
            });
        };
    </script>
</head>
<body>
    {content}
</body>
</html>
"""
    
    return html_template.format(content=html_content)

def main():
    # Read the markdown file
    with open('full_paper_combined.md', 'r') as f:
        md_content = f.read()
    
    # Convert to HTML
    html_content = process_markdown(md_content)
    
    # Write HTML file
    with open('paper_for_review.html', 'w') as f:
        f.write(html_content)
    
    print("HTML file created: paper_for_review.html")
    print("\nTo convert to PDF:")
    print("1. Open paper_for_review.html in your web browser")
    print("2. Press Cmd+P (Mac) or Ctrl+P (Windows/Linux)")
    print("3. Select 'Save as PDF'")
    print("4. Ensure margins are set appropriately (usually 1 inch)")
    print("5. Save the PDF")
    
    # Alternative: Create a simple script for automatic conversion if wkhtmltopdf is available
    import subprocess
    try:
        subprocess.run(['which', 'wkhtmltopdf'], check=True, capture_output=True)
        print("\nAlternatively, you can use wkhtmltopdf:")
        print("wkhtmltopdf --enable-local-file-access paper_for_review.html paper_for_review.pdf")
    except:
        pass

if __name__ == "__main__":
    main()