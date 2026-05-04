# Test conventions

## Structure

- Write all tests as **module-level functions** (`def test_...`). Never group tests inside a class.
- Use **`pytest` fixtures** for all shared tensors and inputs. Fixtures should return a single `torch.Tensor`. Avoid fixtures that return dicts, tuples, or other composite types.
- Fixtures may depend on other fixtures via argument injection — prefer composing small fixtures over writing large monolithic ones.

## Type checking and shape contracts

- Annotate helper functions with **`jaxtyping`** (`Float`, `Int`, `Bool`, etc.) and `@jaxtyped(typechecker=beartype)` so shape contracts are verified at call time.
- Use named dimensions in jaxtyping annotations (e.g. `"B N 3"`, `"N_atoms N_atoms"`) to make shape intent explicit.

## Tensor operations

- Replace all `@` matrix multiplications and `torch.matmul` calls with **`einops.einsum`**. The einsum string makes the contraction axes explicit and self-documenting.
- Use **`einops.rearrange`** instead of `view`, `reshape`, `unsqueeze`, `squeeze`, or `permute`.
- Use **`einops.reduce`** instead of `torch.sum`, `torch.mean`, `torch.max`, etc. when reducing over named axes.
- Use **`einops.repeat`** instead of `expand`, `repeat`, or `tile`.
- Prefer einops operations throughout helper functions so shape contracts interact naturally with jaxtyping annotations.
