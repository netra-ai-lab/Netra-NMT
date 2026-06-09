import sentencepiece as spm

model = spm.SentencePieceProcessor()

model.load("tokenizer/spm_32k.model")

text = "ខ្ញុំទៅសាលារៀន I go to school"

print("INPUT:", text)
print("TOKENS:", model.encode(text, out_type=str))
print("IDS:", model.encode(text, out_type=int))