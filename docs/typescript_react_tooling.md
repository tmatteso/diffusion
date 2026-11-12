# Design: static analysis and style for TypeScript + React

**Status:** Draft, for discussion. Nothing here is implemented yet.

## Context

The Python code in this repo is held to a deliberately strict bar:
basedpyright in `all` mode, ruff with `select = ["ALL"]`, pylint against
the Google style guide, pydoclint, black, xenon, and two custom bans
(`enforce-einops`, `enforce-jaxtyping`). All of them run as pre-commit hooks
and again in CI inside a rebuilt devcontainer.

We are about to start writing React code in TypeScript. This doc proposes a
toolchain and a style guide for that code that hold it to the same bar, and
explains why each piece was chosen.

## Goals

- Every Python quality gate has a TypeScript counterpart of equal
  strictness.
- Same enforcement model as Python: hooks run through `pre-commit` with
  `language: system`, tool versions come from a lockfile, and CI reruns
  everything in a clean container.
- Strict by default, with explicit, documented carve-outs: the same
  philosophy as `select = ["ALL"]`.
- A written style guide (a frontend `CLAUDE.md`) that covers what tools
  cannot check.

## Non-goals

- Choosing a framework (Vite SPA vs. Next.js), a component library, or a
  styling approach. The tooling below works with any of them.
- Changing anything about the Python setup.

## Base style guide

Google publishes a
[TypeScript Style Guide](https://google.github.io/styleguide/tsguide.html).
It is the most defensible base document and is consistent with our use of
the Google Python guide. The points we adopt directly:

- No `any`; use `unknown` and narrow it.
- No default exports; named exports only.
- No `namespace`, no `const enum`.
- No `#private` fields; use TypeScript's `private`.
- No `I` prefix on interfaces; prefer `interface` for object shapes.
- Keep type assertions (`as`, `!`) to a minimum.
- JSDoc on exported symbols, without repeating the types.

Google does not publish a React guide, and its open-source lint config
(`gts`) is much milder than what we want, so we do not use it. The Airbnb
React guide used to fill this gap, but `eslint-config-airbnb` is no longer
well maintained, so we do not build on it either. For React semantics the
authority is the React documentation itself, in particular "Rules of React"
and "You Might Not Need an Effect".

## Proposal: tool mapping

| Python tool | TypeScript equivalent | Notes |
|---|---|---|
| basedpyright `all` | `tsc --noEmit` with `strict` and the extra flags below | The compiler is the type checker. `noImplicitOverride` matches `reportImplicitOverride`. |
| `reportAny`, the `Any` ban | typescript-eslint `no-explicit-any` and `no-unsafe-*`, plus `type-coverage --strict --at-least 100` | `type-coverage` also catches `any` leaking in from third-party types. |
| ruff `select = ["ALL"]` | ESLint (flat config) with typescript-eslint `strictTypeChecked` + `stylisticTypeChecked` | Type-aware linting is the key property. |
| pylint | `eslint-plugin-unicorn`, `eslint-plugin-sonarjs`, `eslint-plugin-import-x` | Deeper semantic and idiom checks. |
| black | Prettier | `eslint-config-prettier` turns off ESLint's formatting rules. |
| xenon / radon | ESLint `complexity`, `max-depth`, `max-params`, `max-lines-per-function`, `sonarjs/cognitive-complexity` | Cognitive complexity tracks readability better than cyclomatic. |
| pydoclint | `eslint-plugin-jsdoc` (TypeScript preset) | Enforces the Google guide's JSDoc rules; see [Documentation linting](#documentation-linting). |
| `enforce-einops`, `enforce-jaxtyping` | `no-restricted-syntax`, `no-restricted-imports`, `no-restricted-properties` | Each ban is one rule with an AST selector; no custom script needed. |
| jaxtyping runtime checks | Zod or Valibot schemas at I/O boundaries | TS types are erased at runtime, so untrusted data must be parsed. |
| (none) | knip | Unused files, exports and dependencies. |
| (none) | dependency-cruiser | Enforces import layering between directories. |
| commitlint | unchanged | |

### Why ESLint and not Biome or Oxlint

Biome and Oxlint are Rust-based and an order of magnitude faster, which is
attractive. But the rules that catch the most real bugs in React code
(`no-floating-promises`, `no-misused-promises`, `strict-boolean-expressions`,
the `no-unsafe-*` family) need type information from the TypeScript
compiler, and typescript-eslint still has the most complete set of them.
Speed can be recovered with `--cache`. If Oxlint's type-aware rules catch up,
it can be added as a fast first pass in front of ESLint.

## Compiler configuration

Start from `@tsconfig/strictest` and make sure these are on:

```jsonc
{
  "compilerOptions": {
    "strict": true,
    "noUncheckedIndexedAccess": true,      // arr[i] is T | undefined
    "exactOptionalPropertyTypes": true,    // `x?: T` is not `x: T | undefined`
    "noImplicitOverride": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "noPropertyAccessFromIndexSignature": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "verbatimModuleSyntax": true,
    "isolatedModules": true,
    "erasableSyntaxOnly": true,            // bans enums, namespaces, param properties
    "allowUnreachableCode": false,
    "skipLibCheck": true
  }
}
```

`noUncheckedIndexedAccess` and `exactOptionalPropertyTypes` are not part of
`strict`. They are the closest equivalent to basedpyright's extra
strictness and catch real bugs: unchecked array access and confusion
between "missing" and "present but undefined".

## Lint configuration

ESLint has no single `select = ["ALL"]` switch, but most plugins ship an
`all` preset (`js.configs.all`, `tseslint.configs.all`,
`unicorn.configs.all`). We propose starting from those and turning rules
off one at a time with a documented reason, as `pyproject.toml` does for
ruff. Some rules in the `all` presets contradict each other, so expect an
initial curation pass. The fallback, if that proves too noisy, is
`strictTypeChecked` plus an explicit list of extra rules.

Rules that must stay on, and why:

- **`@typescript-eslint/no-floating-promises`, `no-misused-promises`:**
  unawaited async calls and async handlers passed to `onClick` are the most
  common real bug class in React code.
- **`@typescript-eslint/strict-boolean-expressions`:** catches
  `{count && <Badge />}`, which renders a literal `0` on screen.
- **`@typescript-eslint/switch-exhaustiveness-check`:** makes discriminated
  unions safe, like `assert_never` in Python.
- **`@typescript-eslint/consistent-type-assertions`
  (`assertionStyle: "never"`) and `no-non-null-assertion`:** ban `as` and
  `!`, the equivalents of `cast()` and `# type: ignore`.
- **`@typescript-eslint/ban-ts-comment`:** bans `@ts-ignore`; requires a
  description on `@ts-expect-error`.
- **`eslint-plugin-react-hooks`** (v6+, which includes the React Compiler
  rules): stale closures, conditional hooks, impure renders.
- **`@eslint-react/eslint-plugin`** (or `eslint-plugin-react`): keys, no
  index-as-key, no components defined inside components.
- **`eslint-plugin-jsx-a11y`:** static accessibility checks.
- **`import-x/no-default-export`:** from the Google guide; named exports
  refactor and grep reliably.
- **`eslint-plugin-testing-library`, `@vitest/eslint-plugin`:** tests query
  by role and text, not implementation details.
- **`@typescript-eslint/explicit-function-return-type`** with
  `allowExpressions: true` and `allowTypedFunctionExpressions: true`: every
  named function, exported or not, declares its return type, like ruff's
  `ANN` rules. Inline callbacks (JSX handlers, `.map` arguments) are exempt,
  because their type comes from context and is never part of an API.
- **`@typescript-eslint/explicit-module-boundary-types`:** exported
  functions, components and hooks also annotate every parameter.
- **`no-nested-ternary`, `unicorn/no-nested-ternary`:** nested ternaries in
  JSX are hard to read; use a named variable or an early return instead.
- **`func-style: ["error", "declaration"]`:** named functions, including
  components, hooks and handlers, are `function` declarations, not
  `const foo = () => …`. This follows the Google guide, reads like a Python
  `def`, and keeps names in stack traces. If we choose
  `eslint-plugin-react`, `react/function-component-definition`
  (`namedComponents: "function-declaration"`) enforces the same for
  components.
- **`prefer-arrow-callback`:** callbacks passed to other functions are
  arrows, not `function` expressions.

ESLint runs with `--max-warnings 0`, so there is no "warning" tier.

## Documentation linting

`eslint-plugin-jsdoc` plays the role of pydoclint. We do not use
`eslint-plugin-tsdoc`: its only rule, `tsdoc/syntax`, checks that comments
parse as TSDoc. It does not require docs to exist, check `@param` names
against the signature, or ban types in docs, and the Google guide does not
use the TSDoc dialect.

How the Google guide's JSDoc rules map to lint rules:

| Google TS guide rule | pydoclint analogue | Lint rule |
|---|---|---|
| `/** */` for docs, `//` for implementation comments | none | `jsdoc/no-bad-blocks`, core `multiline-comment-style: ["error", "separate-lines"]` (skips JSDoc by default) |
| Document all top-level exports | ruff `D1xx` | `jsdoc/require-jsdoc` |
| Don't repeat types already in the signature | `arg-type-hints-in-docstring = false` | `jsdoc/no-types`; the TS preset turns off `require-param-type` and `require-returns-type` |
| Don't use tags TS makes redundant (`@private`, `@override`, `@implements`, `@enum`) | none | `jsdoc/check-tag-names` with `typed: true` |
| `@param` names match the signature | DOC101–DOC105 | `jsdoc/check-param-names` |
| Descriptions add information, not restate the name | none | `jsdoc/informative-docs`, `jsdoc/require-description` |
| Parameter properties are documented with `@param` on the constructor | none | `jsdoc/check-param-names`, `jsdoc/require-param` |
| Returns and raises sections | DOC201+, DOC501+ | `jsdoc/require-returns`, `jsdoc/require-throws` |

Proposed configuration:

```ts
import jsdoc from "eslint-plugin-jsdoc";

export default [
  jsdoc.configs["flat/recommended-typescript-error"],
  {
    rules: {
      "jsdoc/require-jsdoc": ["error", {
        publicOnly: false, // every function, as on the Python side
        require: {
          FunctionDeclaration: true,
          ClassDeclaration: true,
          MethodDefinition: true,
        },
        contexts: ["TSInterfaceDeclaration", "TSTypeAliasDeclaration",
                   "TSPropertySignature"],
      }],
      "jsdoc/check-tag-names": ["error", { typed: true }],
      "jsdoc/no-types": "error",
      "jsdoc/require-param": ["error", { checkDestructured: false }],
      "jsdoc/check-param-names": ["error", { checkDestructured: false }],
      "jsdoc/require-returns": "error",
      "jsdoc/require-throws": "error",
      "jsdoc/require-description": "error",
      "jsdoc/informative-docs": "error",
      "jsdoc/no-blank-blocks": "error",
    },
  },
];
```

One deliberate departure from the Google guide: Google makes `@param` and
`@return` optional, to be written only when they add information. We
require them, matching the full `Args`/`Returns` sections that pydoclint
enforces on the Python side. Following Google strictly would mean turning
off `require-param` and `require-returns` while keeping `check-param-names`,
so tags that are present still cannot drift from the signature.

For React components, `checkDestructured: false` avoids demanding a
`@param props.foo` tag for every destructured prop. Props are documented
on the fields of the `XProps` interface instead, which
`TSPropertySignature` in `require-jsdoc` enforces.

## React conventions for the style guide

These go in a frontend `CLAUDE.md` and play the role of the "never `Any`,
`TypedDict` or `np.generic`" rules on the Python side:

- Function components only.
- Props are a named `interface XProps` with `readonly` fields.
- Derive values during render instead of syncing state in `useEffect`.
- Server data goes through TanStack Query (or similar), not hand-written
  fetch-in-effect code.
- Every external payload (API responses, URL params, storage) is parsed
  with a Zod or Valibot schema.
- UI states are discriminated unions
  (`{status: "loading"} | {status: "error"; error: Error} | {status: "ok"; data: T}`),
  not several independent booleans.
- Event handlers that don't use the event don't accept it:
  `function handleOpen(): void`, passed as `onClick={handleOpen}`.
- When a handler needs one value from the event, read it inline and pass a
  plain value on, e.g.
  `onChange={(event) => { updateName(event.currentTarget.value); }}`.
  The logic then lives in a function such as `updateName(name: string)`
  that can be tested without rendering.
- A component's own callback props are domain-level, e.g.
  `onSelect: (userId: string) => void`, not DOM handler types such as
  `React.MouseEventHandler<HTMLButtonElement>`.
- When a named handler really needs the event (`preventDefault`, keyboard
  handling), annotate the parameter:
  `function handleKeyDown(event: React.KeyboardEvent<HTMLInputElement>): void`.
- Components that wrap a native element extend its props
  (`interface ButtonProps extends React.ComponentProps<"button">`) instead
  of retyping them.
- Conditional rendering goes in named variables or early returns above the
  JSX, not nested ternaries. Every hook call comes before the first early
  return: a `return` above a `useState` or `useEffect` breaks the Rules of
  Hooks.
- Named functions, components, hooks and handlers are `function`
  declarations. Arrows are for short callbacks such as
  `users.map((user) => user.name)`.
- Inline arrows in JSX are only for passing an argument, usually in a
  list: `onClick={() => { onSelect(user.id); }}`. If one grows beyond
  that, move the row into its own component with a named handler.
  `react/jsx-no-bind` stays off, because this case needs inline arrows.

## Testing

- **Vitest** + **React Testing Library** for unit and component tests.
- **MSW** to mock the network at the request level.
- **Playwright** for end-to-end tests, with `@axe-core/playwright` for
  runtime accessibility checks.
- `expectTypeOf` for type-level tests of shared types and hooks.
- Coverage thresholds enforced in `vitest.config.ts`.
- Tests are linted and type-checked with no carve-outs, as `pallatom/tests/`
  is today.

## Enforcement

- **Package manager:** pnpm, with `pnpm-lock.yaml` committed.
- **Pre-commit:** new `language: system` hooks calling `pnpm exec` for
  Prettier, ESLint, `tsc --noEmit`, knip and `type-coverage`, restricted to
  the frontend directory with `files:`. This follows the existing rule that
  tool versions come from the lockfile, not a `rev:` pin. ESLint's typed
  linting builds a full TypeScript program, so it runs once per project
  with `pass_filenames: false` and `--cache`, not once per file.
- **Devcontainer:** add Node and pnpm so CI's rebuilt container can run the
  hooks.
- **CI:** ARM64 CI already runs `pre-commit run --all-files`, so the new
  hooks run there automatically. Add `vitest run` (and later Playwright) as
  a test step alongside `pytest pallatom`. CUDA CI needs no change.

## Rollout

Each step is its own PR. Per repo convention, guide changes are kept out of
tooling PRs.

1. Scaffold the frontend package: `package.json`, `tsconfig.json`,
   `eslint.config.ts`, Prettier config, one trivial component and test.
2. Wire the hooks into `.pre-commit-config.yaml` and add Node to the
   devcontainer.
3. Add the frontend test step to CI.
4. Add a frontend `CLAUDE.md` and update `docs/best_practices.md` and the
   root `CLAUDE.md` hook inventory.

Turning everything on before any real code exists is deliberate: adopting
strict rules on an empty codebase costs nothing, and retrofitting them later
is expensive.

## Open questions

- Where the frontend lives (`frontend/`, `web/`, ...) and whether it needs
  its own devcontainer image.
- Vite SPA or a server-rendering framework. This affects the ESLint
  config's environment, not the tool choices.
- Whether the `all` presets prove workable, or we fall back to
  `strictTypeChecked` plus an explicit list.
- Exact major versions. They should be checked when the scaffold PR is
  written, since this ecosystem moves quickly.
