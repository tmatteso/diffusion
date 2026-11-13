import js from "@eslint/js";
import eslintReact from "@eslint-react/eslint-plugin";
import stylex from "@stylexjs/eslint-plugin";
import vitest from "@vitest/eslint-plugin";
import { defineConfig, globalIgnores } from "eslint/config";
import prettier from "eslint-config-prettier";
import { createTypeScriptImportResolver } from "eslint-import-resolver-typescript";
import checkFile from "eslint-plugin-check-file";
import { importX } from "eslint-plugin-import-x";
import jsdocPlugin from "eslint-plugin-jsdoc";
import jsxA11y from "eslint-plugin-jsx-a11y-x";
import reactHooks from "eslint-plugin-react-hooks";
import { configs as sonarjsConfigs } from "eslint-plugin-sonarjs";
import testingLibrary from "eslint-plugin-testing-library";
import unicorn from "eslint-plugin-unicorn";
import globals from "globals";
import tseslint from "typescript-eslint";

// Every rule in the `all` presets is on, as with ruff's `select = ["ALL"]`.
// Each rule turned off below says why. See
// docs/design/typescript_react_tooling.md for the reasoning behind the
// toolchain.
export default defineConfig(
  globalIgnores(["dist/", "node_modules/"]),

  {
    files: ["**/*.{ts,tsx}"],
    extends: [
      js.configs.all,
      tseslint.configs.all,
      unicorn.configs.all,
      sonarjsConfigs.recommended,
      importX.flatConfigs.recommended,
      importX.flatConfigs.typescript,
      jsdocPlugin.configs["flat/recommended-typescript-error"],
    ],
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    linterOptions: {
      reportUnusedDisableDirectives: "error",
      reportUnusedInlineConfigs: "error",
    },
    settings: {
      "import-x/resolver-next": [createTypeScriptImportResolver()],
    },
    rules: {
      // --- Turned off, with the reason. ---

      // Alphabetical key order hurts readability more than it helps (ruff
      // doesn't sort dict keys either).
      "sort-keys": "off",
      // Replaced by import-x/order below, which groups imports like isort.
      "sort-imports": "off",
      // `undefined` is everywhere in strict TypeScript (optional properties,
      // unchecked index access). The alternative, `void 0`, is worse.
      "no-undefined": "off",
      // Google-style JSDoc uses ` * ` prefixes, which this rule bans.
      "unicorn/no-asterisk-prefix-in-documentation-comments": "off",
      // Prettier doesn't wrap comments, so keeping them within 80 columns
      // means wrapping them by hand.
      "unicorn/no-manually-wrapped-comments": "off",
      // Tsc already checks these, and type-aware linting makes them slow.
      // typescript-eslint's docs recommend turning them off.
      "import-x/default": "off",
      "import-x/named": "off",
      "import-x/namespace": "off",
      "import-x/no-named-as-default-member": "off",
      "import-x/no-unresolved": "off",

      // --- Reconfigured. ---

      // Components are PascalCase files, everything else camelCase
      // (sampling_website.md §6.1). check-file enforces which goes where.
      "unicorn/filename-case": [
        "error",
        { cases: { camelCase: true, pascalCase: true } },
      ],
      // A blank line between the description and the tags, as in the Google
      // guide.
      "jsdoc/tag-lines": ["error", "never", { startLines: 1 }],
      // Wrapped comments continue mid-sentence on the next line.
      "capitalized-comments": [
        "error",
        "always",
        { ignoreConsecutiveComments: true },
      ],
      // One variable per declaration (Google guide).
      "one-var": ["error", "never"],
      // Groups like isort: Node built-ins, packages, then local files.
      "import-x/order": [
        "error",
        {
          groups: ["builtin", "external", ["parent", "sibling", "index"]],
          "newlines-between": "always",
          alphabetize: { order: "asc", caseInsensitive: true },
        },
      ],

      // --- Rules the tooling doc requires, configured explicitly. ---
      "@typescript-eslint/consistent-type-assertions": [
        "error",
        { assertionStyle: "never" },
      ],
      "@typescript-eslint/explicit-function-return-type": [
        "error",
        { allowExpressions: true, allowTypedFunctionExpressions: true },
      ],
      "func-style": ["error", "declaration"],
      "import-x/no-default-export": "error",
      "multiline-comment-style": ["error", "separate-lines"],

      // Naming (sampling_website.md §6.1). Functions may be PascalCase
      // because components are functions.
      "@typescript-eslint/naming-convention": [
        "error",
        { selector: "typeLike", format: ["PascalCase"] },
        { selector: "function", format: ["camelCase", "PascalCase"] },
        {
          selector: "variable",
          format: ["camelCase", "UPPER_CASE"],
          leadingUnderscore: "allow",
        },
      ],

      // --- JSDoc (tooling doc, "Documentation linting"). ---
      "jsdoc/require-jsdoc": [
        "error",
        {
          publicOnly: false,
          require: {
            FunctionDeclaration: true,
            ClassDeclaration: true,
            MethodDefinition: true,
          },
          contexts: [
            "TSInterfaceDeclaration",
            "TSTypeAliasDeclaration",
            "TSPropertySignature",
          ],
        },
      ],
      "jsdoc/check-tag-names": ["error", { typed: true }],
      "jsdoc/no-types": "error",
      "jsdoc/require-param": ["error", { checkDestructured: false }],
      "jsdoc/check-param-names": ["error", { checkDestructured: false }],
      "jsdoc/require-returns": "error",
      "jsdoc/require-throws": "error",
      "jsdoc/require-description": "error",
      "jsdoc/informative-docs": "error",
      "jsdoc/no-blank-blocks": "error",
      "jsdoc/no-bad-blocks": "error",
    },
  },

  // React code: everything under src/.
  {
    files: ["src/**/*.{ts,tsx}"],
    extends: [
      eslintReact.configs.all,
      eslintReact.configs["disable-rsc"],
      reactHooks.configs.flat["recommended-latest"],
      jsxA11y.configs.strict,
    ],
    languageOptions: {
      globals: globals.browser,
    },
    plugins: {
      // Only the rules used below. The plugin's `valid-shorthands` rule
      // declares an invalid `meta.type`, so the whole plugin object doesn't
      // type-check as an ESLint plugin.
      "@stylexjs": {
        rules: {
          "no-unused": stylex.rules["no-unused"],
          "valid-styles": stylex.rules["valid-styles"],
        },
      },
      "check-file": checkFile,
    },
    rules: {
      "@stylexjs/valid-styles": "error",
      "@stylexjs/no-unused": "error",

      // Component files are PascalCase, hook files are use* camelCase
      // (sampling_website.md §6.1).
      "check-file/filename-naming-convention": [
        "error",
        {
          "src/components/**/*.tsx": "PASCAL_CASE",
          "src/hooks/**/*.ts": "use+([a-zA-Z0-9])",
        },
        { ignoreMiddleExtensions: true },
      ],
    },
  },

  // Tests.
  {
    files: ["src/**/*.test.{ts,tsx}"],
    extends: [vitest.configs.all, testingLibrary.configs["flat/react"]],
    settings: {
      // Lets vitest/valid-title accept `describe(App, …)`.
      vitest: { typecheck: true },
    },
    rules: {
      // A synchronous test can't finish before its assertions run, so only
      // async tests need expect.assertions().
      "vitest/prefer-expect-assertions": [
        "error",
        { onlyFunctionsWithAsyncKeyword: true },
      ],
    },
  },

  // Tool configuration files run in Node, and Vite and ESLint require a
  // default export from them.
  {
    files: ["*.config.ts"],
    languageOptions: {
      globals: globals.node,
    },
    rules: {
      "import-x/no-default-export": "off",
    },
  },

  // Last, so Prettier owns formatting.
  prettier,
);
