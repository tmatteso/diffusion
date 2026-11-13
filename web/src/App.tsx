import * as stylex from "@stylexjs/stylex";

const styles = stylex.create({
  root: {
    fontFamily: "system-ui, sans-serif",
    minHeight: "100vh",
  },
});

/**
 * Root component of the sampling website.
 *
 * @returns The page shell.
 */
export function App(): React.JSX.Element {
  return (
    <main {...stylex.props(styles.root)}>
      <h1>pallatom sampler</h1>
    </main>
  );
}
