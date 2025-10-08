from train_pipe import *
import sys
import lxml.etree as ET

def parse_xml(file):
    '''
    Parse the XML file and return the segments.
    Args:
        file: path to the XML file
    Returns:
        segments: list of segments
    '''
    try:
        tree = ET.parse(file,  ET.XMLParser(recover=True))
    except IOError:
        raise IOError(f"File {file} not found.")
    except SyntaxError:
        raise SyntaxError(f"File {file} is not well-formed.")

    root = tree.getroot()
    segments = []
    for doc in root: #tree.getiterator(tag='DOC'):
        for tag in doc:
            if tag.tag == 'SEG':
                if tag.text is None:
                    segments.append('')
                else:
                    segments.append(tag.text.strip())
    return segments

class SGMDATASET(Dataset):
    def __init__(self, source, tokenizer, src_max_length):
        self.source = [f"Translate the following English sentence into German:\n{seg} <de>" for seg in parse_xml(source)]
        self.tokenizer = tokenizer
        self.src_max_length = src_max_length

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src_text = self.source[idx] + self.tokenizer.eos_token
        enc_src = self.tokenizer(
            src_text,
            truncation=True,
            max_length=self.src_max_length,
            padding="max_length",
            return_tensors="pt"
        )
        src_input_ids = enc_src.input_ids.squeeze(0)
        src_attention_mask = enc_src.attention_mask.squeeze(0)

        src_len = (src_input_ids != self.tokenizer.pad_token_id).sum().item()
        return {
            'src_input_ids': src_input_ids,
            'src_attention_mask': src_attention_mask,
            'src_len': src_len
        }

if len(sys.argv) != 3:
    print("Usage: python translate.py <source_file> <output_file>")
    sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained('utter-project/EuroLLM-1.7B', padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
model = TranslationFineTuner('utter-project/EuroLLM-1.7B')
model.eval()
model.freeze()
source_data = SGMDATASET(sys.argv[1], tokenizer, 128)
dataloader = DataLoader(source_data, batch_size=4, shuffle=False, num_workers=2)
trainer = pl.Trainer()
predictions = trainer.predict(model, dataloaders=dataloader)
predictions = [pred for batch in predictions for pred in batch]
predictions = [pred[len(seg):] for seg, pred in zip(source_data.source, predictions)]

out = ET.parse(sys.argv[1],  ET.XMLParser(recover=True))
for doc in out.getroot(): #tree.getiterator(tag='doc'):
    for tag in doc:
        if tag.tag == 'SEG':
            tag.text = predictions.pop(0)
out.write(sys.argv[2], encoding='utf-8', xml_declaration=True)