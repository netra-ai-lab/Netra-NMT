from pathlib import Path
import sentencepiece as spm

print("\nSCRIPT STARTED")

ROOT = Path(__file__).resolve().parent.parent

corpus = ROOT / "tokenizer" / "corpus.txt"
out_dir = ROOT / "tokenizer"
out_dir.mkdir(exist_ok=True)

print("Corpus path:", corpus)

configs = [
    ("spm_32k", 32000),
    ("spm_48k", 48000),
]

print("Training SentencePiece tokenizers...")

for name, vocab_size in configs:

    print("\n" + "=" * 60)
    print(f"Training: {name}")
    print("=" * 60)

    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(out_dir / name),

        vocab_size=vocab_size,
        model_type="unigram",

        character_coverage=0.9995,

        # IMPORTANT: bilingual direction tokens
        user_defined_symbols=[
            "<2en>",
            "<2km>"
        ],

        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,

        input_sentence_size=1000000,
        shuffle_input_sentence=True,

        # logging
        train_extremely_large_corpus=True
    )

    print(f"Finished: {name}")

print("\nDONE: tokenizer saved in tokenizer/")