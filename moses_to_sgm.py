import lxml.etree as ET
import sys

# input_name = 'demo/baseline'
# output_name = 'demo/baseline1.sgm'

# with open(input_name, 'r', encoding='utf-8') as f:
#     lines = f.readlines()

# root = ET.Element('tstset', attrib={'trglang':'fr', 'setid':'test','srclang':'any'})
# doc = ET.SubElement(root, 'doc', attrib={'docid':'test'})
# for i, line in enumerate(lines):
#     seg = ET.SubElement(doc, 'seg', id=str(i+1))
#     seg.text = line.strip()
# #save root to file
# tree = ET.ElementTree(root)
# tree.write(output_name, encoding='utf-8', xml_declaration=False, pretty_print=True)
for filename in sys.argv[1:]:
    try:
        tree = ET.parse(filename)
        root = ET.Element('tstset', attrib={'trglang':'fr', 'setid':'test','srclang':'any'})
        doc = ET.SubElement(root, 'DOC', attrib={'Docid':'test'})
        for i, seg in enumerate(tree.getroot()[0]): #tree.getiterator(tag='doc'):
            if seg.tag == 'seg':
                new_seg = ET.SubElement(doc, 'SEG', id=str(i+1))
                new_seg.text = seg.text
        #save root to file
        tree = ET.ElementTree(root)
        tree.write(filename, encoding='utf-8', xml_declaration=False, pretty_print=True)
    except Exception as e:
        print(f"Error processing file {filename}: {e}")