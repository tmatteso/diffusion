// Extends Conventional Commits with cbea.ms formatting rules.
// https://www.conventionalcommits.org/  https://cbea.ms/git-commit/

module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    // Override conventional's 100-char default to 72 (cbea.ms rule 2)
    'header-max-length': [2, 'always', 72],

    // Disable case enforcement — conventional defaults to lower-case, we don't care
    'subject-case': [0],

    // cbea.ms rule 4: no trailing period on subject line
    'subject-full-stop': [2, 'never', '.'],

    // cbea.ms rule 6: wrap body at 72 characters
    'body-max-line-length': [2, 'always', 72],

    // cbea.ms rule 1: blank line between subject and body (also in conventional, made explicit)
    'body-leading-blank': [2, 'always'],
  },
};
