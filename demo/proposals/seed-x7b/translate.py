from vllm import LLM
from vllm.sampling_params import SamplingParams, BeamSearchParams
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
model = LLM(model=args.model,
            max_num_seqs=512,
            tensor_parallel_size=1,
            enable_prefix_caching=False, #True, 
            gpu_memory_utilization=0.95,
            dtype='half')

messages = [
    f"Translate the following English sentence into German and explain it in detail:\n{seg} <de>" for seg in source 
]
# Greedy decoding
decoding_params = SamplingParams(temperature=0,
                                 max_tokens=512,
                                 skip_special_tokens=True)

results = model.generate(messages, decoding_params)
predictions = []
for res in results:
    res = res.outputs[0].text.strip()
    res = res.split('[COT]')[0].strip()  # Remove CoT explanation if present
    predictions.append(res)
##################
# WRITE OUTPUT FILE
##################
out = ET.parse(args.source,  ET.XMLParser(recover=True))
for doc in out.getroot():
    for tag in doc:
        if tag.tag == 'SEG':
            tag.text = predictions.pop(0)
out.write(args.output, encoding='utf-8', xml_declaration=True)