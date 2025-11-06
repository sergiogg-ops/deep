import lxml.etree as ET
import argparse
import os

def parse_xml(file):
    try:
        tree = ET.parse(file,  ET.XMLParser(recover=True))
    except IOError:
        raise IOError(f"File {file} not found.")
    except SyntaxError:
        raise SyntaxError(f"File {file} is not well-formed.")
    root = tree.getroot()
    
    # Find all page elements
    pages = root.findall('page')

    data = {}
    for page in pages:
        doc_id = page.get('doc')
        if doc_id not in data:
            data[doc_id] = [page]
        else:
            data[doc_id].append(page)

    return data

def split_pages(docs, data, dir='output/'):
    for doc_id in docs:
        documents = data.get(doc_id)
        if documents:
            new_root = ET.Element('archer-ocr')
            for doc in documents:
                new_root.append(doc)
            new_tree = ET.ElementTree(new_root)
            output_dir = os.path.join(os.getcwd(), doc_id)
            os.makedirs(output_dir, exist_ok=True)
            print(f'Saving document ID {doc_id} with {len(documents)} pages to {output_dir}/{dir}.xml')
            new_tree.write(f'{output_dir}/{dir}.xml', pretty_print=True, xml_declaration=True, encoding='UTF-8')

def parse_args():
    parser = argparse.ArgumentParser(description='Split XML pages by document IDs.')
    parser.add_argument('input_file', type=str, help='Path to the input XML file.')
    parser.add_argument('--system', type=str, default='output/', help='Directory to save the split XML files.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    input_file = args.input_file
    docs_to_split = ['AIDWLU', 'AIGDTW','AJBJWO','ARBXCK']  # Replace with your document IDs to split
    system = args.system

    data = parse_xml(input_file)
    split_pages(docs_to_split, data, dir=system)