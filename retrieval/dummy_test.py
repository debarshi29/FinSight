import fitz

doc = fitz.open("..\Long_Term_Memory_Flow.pdf")

for page_num in range(len(doc)):
    page = doc[page_num]
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        if block["type"] != 0:
            continue
        print(block)
        print("---")
