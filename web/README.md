# web/: sampling website frontend

React 19 + TypeScript, built with Vite and deployed to GitHub Pages. See
[`docs/design/sampling_website.md`](../docs/design/sampling_website.md) for
the design and
[`docs/design/typescript_react_tooling.md`](../docs/design/typescript_react_tooling.md)
for why the toolchain is set up this way.

## Commands

Run everything from `web/`. Node 22 and pnpm come with the devcontainer.

| Command                             | What it does                                          |
| ----------------------------------- | ----------------------------------------------------- |
| `pnpm install --frozen-lockfile`    | Install the exact versions in `pnpm-lock.yaml`        |
| `pnpm dev`                          | Dev server with hot reload                            |
| `pnpm build`                        | Type-check, then build to `dist/`                     |
| `pnpm test`                         | Vitest, once                                          |
| `pnpm typecheck`                    | `tsc --build` over the app and the config files       |
| `pnpm lint`                         | ESLint with `--max-warnings 0`                        |
| `pnpm format:check` / `pnpm format` | Prettier check / rewrite                              |
| `pnpm type-coverage`                | Fails on any `any`, including ones from library types |
| `pnpm knip`                         | Unused files, exports and dependencies                |

## Layout

- `tsconfig.app.json` covers `src/` (browser code and tests).
  `tsconfig.node.json` covers the config files, which run in Node.
  `tsconfig.json` only references the two.
- `eslint.config.ts` starts from every rule in the `all` presets. Each rule it
  turns off or reconfigures has a comment saying why.

## Dependencies

Every version is pinned exactly. `pnpm-workspace.yaml` refuses any package
version published less than two weeks ago (`minimumReleaseAge`), so a new
release can't be installed until it's two weeks old.

## Choices made while scaffolding

These differ from, or fill gaps in, the design docs.

- **StyleX through `@stylexjs/unplugin`.** `@vitejs/plugin-react` 6 no longer
  takes Babel plugins, and `@stylexjs/unplugin` is the Vite integration the
  StyleX docs recommend. In dev it injects the CSS into the page itself. In a
  build it appends to the bundle's CSS asset, which is one reason
  `src/styles/reset.css` exists: without an imported stylesheet the StyleX
  CSS would be written to a file nothing links.
  - `vite.config.ts` drops the plugin's dev-server hook under Vitest, because
    the hook starts a timer that would keep Vitest from exiting.
  - `patches/` types the plugin's `vite()` as returning Vite's `Plugin`
    instead of `any`. Drop the patch once upstream fixes the types.
- **TypeScript 6, not 7.** typescript-eslint's type-aware rules support
  TypeScript `<6.1`.
- **`@eslint-react/eslint-plugin` instead of `eslint-plugin-react`, and
  `eslint-plugin-jsx-a11y-x` instead of `eslint-plugin-jsx-a11y`.** The
  originals don't support ESLint 10. `@eslint-react`'s `no-class-component`
  bans class components; `tsc` rejects lowercase component names in JSX,
  which is what `react/jsx-pascal-case` would have caught.
