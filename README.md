# bert_create_pretraining

\[ [API doc](https://docs.rs/bert_create_pretraining/) | [crates.io](https://crates.io/crates/bert_create_pretraining/) \]

The crate provides the port of the original BERT create_pretraining_data.py script from the [Google BERT repository](https://github.com/google-research/bert).

## Usage

```bash
find "${DATA_DIR}" -name "*.txt" | xargs -I% -P $NUM_PROC -n 1 \
basename % | xargs -I% -P ${NUM_PROC} -n 1 \
  "${TARGET_DIR}/bert_create_pretraining" \
  --input-file="${DATA_DIR}/%" \
  --output-file="${OUTPUT_DIR}/%.tfrecord" \
  --vocab-file="${VOCAB_DIR}/vocab.txt" \
  --max-seq-length=512 \
  --max-predictions-per-seq=75 \
  --masked-lm-prob=0.15 \
  --random-seed=12345 \
  --dupe-factor=5
```

## License

MIT license. See [LICENSE](LICENSE) file for full license.
