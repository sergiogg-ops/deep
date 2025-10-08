from transformers import TranslationPipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from argparse import ArgumentParser
import lxml.etree as ET

parser = ArgumentParser(description='Translate the source file using a pretrained model.')
parser.add_argument('-model', type=str, required=True, help='Path to the pretrained model')
parser.add_argument('-source', type=str, required=True, help='Path to the source file')
parser.add_argument('-output', type=str, required=True, help='Path to the output file')
parser.add_argument('-src_lang', type=str, default='en_XX', help='Source language code')
parser.add_argument('-tgt_lang', type=str, default='fr_XX', help='Target language code')
parser.add_argument('-batch_size', type=int, default=8, help='Batch size for translation')
args = parser.parse_args()

##################
# READ INPUT FILE
##################
try:
    tree = ET.parse(args.source,  ET.XMLParser(recover=True))
except IOError:
    raise IOError(f"File {args.source} not found.")
except SyntaxError:
    raise SyntaxError(f"File {args.source} is not well-formed.")

root = tree.getroot()
source = []
for doc in root:
    for tag in doc:
        if tag.tag == 'SEG':
            if tag.text is None:
                source.append('')
            else:
                source.append(tag.text.strip())

##################
# TRANSLATE
##################
tokenizer = AutoTokenizer.from_pretrained(args.model, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model).eval()

translation_pipeline = TranslationPipeline(model=model, tokenizer=tokenizer, batch_size=args.batch_size, num_workers=2, device='cuda')
predictions = translation_pipeline(source, max_length=512, num_beams=4, early_stopping=True, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
predictions = [pred['translation_text'] for pred in predictions]

##################
# WRITE OUTPUT FILE
##################
out = ET.parse(args.source,  ET.XMLParser(recover=True))
for doc in out.getroot():
    for tag in doc:
        if tag.tag == 'SEG':
            tag.text = predictions.pop(0)
out.write(args.output, encoding='utf-8', xml_declaration=True)