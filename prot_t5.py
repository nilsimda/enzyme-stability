#usr/bin/env python3

import torch
import re
import sys
import os

from tqdm import tqdm
from Bio import SeqIO
from transformers import T5Tokenizer, T5EncoderModel

def preprocess_from_fasta(path):
    labels, seqs = [], []
    for record in SeqIO.parse(path, "fasta"):
        labels.append(record.id)

        seq = str(record.seq)[:1022]
        seq = re.sub(r'[JUZOB\*]', 'X', seq)
        seqs.append(' '.join(list(seq)))

    return labels, seqs
        


def main(p_dir, t5_dir):
    virus_fastas = os.listdir(p_dir)
    no_prots = 0
    batch_size=1

    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    prott5 = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float16)
    prott5.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prott5 = prott5.to(device)

    for count, fasta in enumerate(virus_fastas, 1):
        prot_embs = []
        labels, seqs = preprocess_from_fasta(f"{p_dir}/{fasta}")
        if len(labels) == 0:
            no_prots+=1
            continue

        pbar = tqdm(range(0, len(labels), batch_size), position=0, leave=False)
        pbar.set_description(f"{count}/{len(virus_fastas)}")

        for i in pbar:
            ids = tokenizer.batch_encode_plus(seqs[i:i+batch_size], add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            with torch.no_grad():
                results = prott5(input_ids=input_ids, attention_mask=attention_mask)

            per_residue = results.last_hidden_state.detach().cpu()
            prot_embs.append(torch.mean(per_residue, dim=1))

        prot_embs = list(torch.cat(prot_embs))
        file_name = ".".join([fasta.split(".")[0], "t5", "pt"])
        torch.save(dict(zip(labels, prot_embs)), f"{t5_dir}/{file_name}")
    print(f"No proteins for {no_prots} Viruses")

if __name__ == '__main__':
    p_dir, t5_dir = sys.argv[1], sys.argv[2]
    print(f"Reading fastas from {p_dir}")
    print(f"Writing embeddings to {t5_dir}")
    main(p_dir, t5_dir)

