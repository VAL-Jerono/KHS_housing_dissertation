import json

with open("KHS_Dissertation_Final_Enhanced.ipynb", "r") as f:
    nb = json.load(f)

with open("current_markdowns.txt", "w") as out:
    for i, cell in enumerate(nb.get("cells", [])):
        if cell["cell_type"] == "markdown":
            source = "".join(cell.get("source", []))
            # Just taking the first line or so to identify it, or full text if short
            out.write(f"\n--- Cell {i} (Markdown) ---\n")
            out.write(source + "\n")
        elif cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            out.write(f"\n--- Cell {i} (Code) ---\n")
            out.write(source[:100].replace('\n', ' ') + "...\n")
