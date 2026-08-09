# IRCoT attribution

The IRCoT control flow in `src/agent/ircot_agent.py` is a reimplementation of
code from [StonyBrookNLP/ircot](https://github.com/StonyBrookNLP/ircot) at
commit `3c1820f698eea5eeddb4fba3c56b64c961e063e4`.

The optional prompt assets downloaded by `scripts/fetch_ircot_assets.py` are
unmodified files from that same commit. IRCoT is distributed under the
Apache License 2.0. The upstream copyright and license remain applicable.

This repository substitutes its configured local model (such as Qwen) for
the upstream `code-davinci-002` generator. It therefore reproduces the IRCoT
method, not the original model outputs.
