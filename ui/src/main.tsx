import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { completeSignInIfRedirected } from "./auth";
import { Root } from "./ChatApp";
import "./app.css";

async function start(): Promise<void> {
  let signInError: string | undefined;
  try {
    await completeSignInIfRedirected();
  } catch (err) {
    signInError = err instanceof Error ? err.message : "Sign-in failed.";
  }
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <Root signInError={signInError} />
    </StrictMode>,
  );
}

void start();
