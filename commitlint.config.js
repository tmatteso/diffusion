// @ts-check
// Rules derived from https://cbea.ms/git-commit/

/** @type {import('@commitlint/types').UserConfig} */
module.exports = {
  rules: {
    // Rule 1: Separate subject from body with a blank line
    'body-leading-blank': [2, 'always'],

    // Rule 2: Limit the subject line to 72 characters (article recommends 50 as a soft target)
    'header-max-length': [2, 'always', 72],

    // Rule 3: Capitalize the subject line
    'subject-case': [2, 'always', 'sentence-case'],

    // Rule 4: Do not end the subject line with a period
    'subject-full-stop': [2, 'never', '.'],

    // Rule 5: Use the imperative mood — not automatically enforceable by commitlint

    // Rule 6: Wrap the body at 72 characters
    'body-max-line-length': [2, 'always', 72],

    // Rule 7: Use the body to explain what and why vs. how — not automatically enforceable
  },
};
