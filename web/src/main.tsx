import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import "./styles/reset.css";

const rootElement = document.querySelector("#root");
if (rootElement === null) {
  throw new Error("index.html has no #root element.");
}

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
