*This project has been created as part of the 42 curriculum by jesequie.*

# call me maybe

Function calling with a small language model (`Qwen/Qwen3-0.6B`) using
**constrained decoding**.

## Description

Given a natural-language request such as *"What is the sum of 265 and 345?"*,
the program does not answer it. It translates it into a structured function
call:

```json
{
  "prompt": "What is the sum of 265 and 345?",
  "name": "fn_add_numbers",
  "parameters": {"a": 265.0, "b": 345.0}
}
```

A 0.6B model asked to "output JSON" in plain text produces invalid JSON often.
Here the model is never trusted to produce valid syntax. At every step the
program computes which tokens keep the output valid under the declared
function schemas, masks all the others, and lets the model choose only among
the valid ones. The output is therefore valid JSON that matches a declared
function by construction. The model decides *content* (which function, which
values); the program decides *validity* (structure, names, types).

## Instructions

Requirements: Python 3.10+, [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                 # install dependencies (the model downloads on first run)
uv run python -m src    # run with the default input/output paths
make lint               # flake8 + mypy
```

validating make run:`verifies that the project had more than 90%`
```bash
cd /sgoinfre/jesequie/42/call_me_maybe/moulinette
uv run python -m moulinette grade_student_answers --set public --student_answer_path /sgoinfre/jesequie/42/call_me_maybe/data/output/function_calls.json
```

The first run downloads the model weights from the Hugging Face Hub (cached
afterwards). Errors (missing file, invalid JSON, wrong structure, empty
function list) are reported as `Error: ...` on stderr with exit code 1.

### Example usage

```bash
echo '[{"prompt": "Greet shrek"}]' > /tmp/in.json
uv run python -m src --input /tmp/in.json --output /tmp/out.json
cat /tmp/out.json
```

## Algorithm: constrained decoding

For each prompt the pipeline (`src/pipeline.py`) runs two phases. Each phase
uses the same loop (`src/decoder.py`):

1. Run the model on the prompt plus the tokens generated so far and read the
   logits of the next token.
2. Ask the constraint (`src/constraints.py`) for the set of **allowed token
   ids**: tokens whose bytes, appended to the text so far, still form a valid
   prefix of a call.
3. Set the logits of every other token to `-inf` and take the `argmax`.
4. Repeat until the call is complete.

**What "valid prefix" means.** `src/schema_parser.py` is an incremental
parser over bytes. It knows the shape
`{"name": "<function>", "parameters": {...}}`, the declared function names,
each parameter name in order, and each parameter type: `number` (a JSON
float), `integer`, `string` (with JSON escape rules and complete UTF-8),
`boolean`, and enums. A candidate token is allowed only if the parser accepts
the text with that token appended.

**Vocabulary.** `src/vocabulary.py` reads the SDK vocabulary and tokenizer
files and builds, for every token id, the exact bytes it stands for (inverting
the GPT-2 byte-to-unicode mapping). All checks work on bytes, not on decoded
strings, so partial UTF-8 characters are handled correctly.

**Phases.**
- *Phase A* shows the model all function definitions and generates until the
  function name is determined.
- *Phase B* shows only the chosen function's definition and continues from the
  generated name to fill in `parameters`. The model still chooses the
  function; the second prompt is just shorter.

**Output.** The generated text is parsed with `json.loads` and validated with
pydantic (`src/models.py`) before being written. `number` parameters are
written as floats (`2.0`), `integer` as ints.

## Design decisions

- **Constraint by bytes, with a parser, not a regex.** Schemas vary by
  function; an incremental parser handles any declared function without
  per-function code.
- **No KV cache in the SDK.** Each forward pass reprocesses the whole context
  on CPU, so cost grows with context length. The performance work therefore
  reduces *context size* and *number of forwards* rather than the cost of one
  forward.
- **Plain-token fast path.** Inside a string value, tokens that contain no
  quote, backslash, control byte and are valid UTF-8 are always allowed. They
  are classified once at start-up, so the parser only runs for the few special
  tokens. An offline check against the old behaviour (every token parsed)
  showed 0 differences on the real vocabulary and cut constraint time from
  about 51 s to 17 s on the checked run.
- **Skip the forward when only one token is allowed** (e.g. fixed JSON
  punctuation), and force the single valid continuation of fixed text such as
  `", "parameters": {"`. The opening quote of a string value is deliberately
  left to the model, because forcing it changes how the tokenizer splits the
  text and degraded the model's output (a leading `/` in paths was lost).
- **Two phases** to shrink the prompt (see Performance).
- **Pydantic everywhere data crosses a boundary** (inputs, definitions,
  results).
- **Thin SDK wrapper** (`src/llm.py`) using only public SDK methods.
  The student code does not import `torch` or `transformers`.

## Performance analysis

| Version | Private set | Notes |
|---|---|---|
| Baseline | 7:45 | every token parsed, one long prompt |
| Fast constraint + two phases | 4:41 | |
| + forced tokens / skipped forwards | 3:37 | |

Current state (CPU, same machine):

| Set | Accuracy | Time |
|---|---|---|
| Public (11 tests) | 10/11 (90.9%) | 3:49 |

- **Reliability:** output is 100% parseable JSON by construction. Decoding is
  deterministic (`argmax`), so repeated runs give the same file.
- **Accuracy limits:** the model is small and sensitive to prompt wording.
  The one public miss is a regex request where the model copies a literal
  (`34`) instead of writing a pattern (`\d+`). Constrained decoding guarantees
  a valid *form*, not a correct *value*.

## Challenges faced

- **Speed.** The first version took 7:45. Profiling showed the constraint
  check dominated; the plain-token fast path and the two-phase prompt brought
  it under the 5-minute limit.
- **Tokenizer boundaries.** Forcing text that ends in a lone `"` token changed
  the model's next-token distribution and broke path arguments. Fixed by
  leaving the opening quote of strings to the model.
- **Prompt sensitivity.** Changes that fixed one case (a template string whose
  label must be dropped) broke another (regex generation). Prompts are now
  validated against both the public and private moulinette sets.

## Testing strategy

- The provided `moulinette` on both the public and the private sets.
- A byte-level equivalence check of the optimized constraint against the
  original one on the real vocabulary.
- A table of parser cases (valid and invalid prefixes for each type).
- Error cases: malformed JSON, missing files, empty function list, wrong
  structure.
- `flake8` and `mypy` with the flags from the subject (`make lint`).

## Resources

- Hugging Face `transformers` and tokenizer documentation.
- Qwen3 model card: <https://huggingface.co/Qwen/Qwen3-0.6B>
- Pydantic documentation: <https://docs.pydantic.dev>
- `uv` documentation: <https://docs.astral.sh/uv/>

### Use of AI

AI (Claude) was used as a tutor and reviewer: explaining constrained decoding
and tokenization before each piece of code, reviewing designs, helping to
diagnose performance and accuracy regressions, and drafting this README. The
code was written step by step by the author, who is able to explain it.
