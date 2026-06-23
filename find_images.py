import json
import glob
import re

notebooks = [
    "KHS_Dissertation_Final_Enhanced.ipynb",
    "EDA_WithAnAngle.ipynb",
    "exploration.ipynb",
    "KHS_EDA_cells.ipynb"
]

for nb_file in notebooks:
    try:
        with open(nb_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        continue
        
    print(f"\n========== {nb_file} ==========")
    for cell in data.get('cells', []):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            
            # Find savefig
            saves = re.findall(r'savefig\([\'"]([^\'"]+)[\'"]', source)
            if saves:
                for save in saves:
                    # Try to find a title nearby to understand what it is
                    title_match = re.search(r'title\([\'"]([^\'"]+)[\'"]', source)
                    title = title_match.group(1) if title_match else "No explicit title found"
                    print(f"Image saved as: {save}")
                    print(f"  Title/Context: {title}")
            
            # Also just look for titles in case savefig isn't used
            elif 'plt.title' in source or 'fig.suptitle' in source or 'set_title' in source:
                titles = re.findall(r'title\([\'"]([^\'"]+)[\'"]', source)
                for t in titles:
                    print(f"Plot generated (not saved): {t}")

