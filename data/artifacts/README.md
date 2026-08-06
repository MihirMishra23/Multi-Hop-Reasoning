# Derived evaluation artifacts

`popqa_corpus_1000_ex_seed_42.json.gz` is a gzip-compressed, byte-preserving copy
of the team's historical PopQA Wikipedia context artifact:

- uncompressed SHA-256: `00c9c266728f63ba0fc259d727ccab488a9ad54bc594eabd10ffe131ca0875fc`
- uncompressed bytes: `10,952,628`
- compressed SHA-256: `7872be316c22a65665ed32ab65f4ca490939a14bd7f204866db3090144a7c995`

The artifact contains Wikipedia article text identified by article title. It is
derived data rather than part of PopQA. Wikipedia text is available under the
[Creative Commons Attribution-ShareAlike License](https://creativecommons.org/licenses/by-sa/4.0/);
the corresponding source page for an entry titled `T` is
`https://en.wikipedia.org/wiki/T`.

Despite its filename, this historical file contains 992 unique articles. The
seed-42 sample contained 1,000 question rows, six repeated subject titles, and
two titles without a saved article. Its article order does not match the PopQA
row order, so loaders must join it by title. Use `scripts/build_popqa_corpus.py`
to produce an auditable full corpus; generated full snapshots should be hosted
outside Git and pinned by commit plus checksum.
