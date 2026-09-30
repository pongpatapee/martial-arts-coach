import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { api } from "./api/client";
import { JobPage, LibraryPage, UploadPage } from "./components/Pages";
import { navigate, useLoad, useRoute, useTheme, type Theme } from "./hooks/common";
import "./styles.css";

function App() {
  const route = useRoute();
  const [theme, setTheme] = useTheme();
  const options = useLoad(api.options, "options");
  const skeleton = useLoad(api.skeleton, "skeleton");

  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="#/">
          <span className="brand-mark" aria-hidden />
          Sparring Analysis
        </a>
        <nav>
          <a href="#/" className={route.page === "library" ? "on" : ""}>
            Library
          </a>
          <a href="#/new" className={route.page === "new" ? "on" : ""}>
            New analysis
          </a>
        </nav>
        <span className="grow" />
        <select
          className="theme"
          value={theme}
          onChange={(e) => setTheme(e.target.value as Theme)}
          aria-label="Theme"
        >
          <option value="system">System theme</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </header>
      <main>
        {options.error ? (
          <div className="page notice error">
            Can't reach the analysis server: {options.error.message}.{" "}
            <button className="btn small" onClick={options.reload}>
              Retry
            </button>
          </div>
        ) : !options.data ? (
          <div className="page muted">Loading…</div>
        ) : route.page === "new" ? (
          <UploadPage options={options.data} />
        ) : route.page === "job" ? (
          <JobPage key={route.id} id={route.id} options={options.data} skeleton={skeleton.data} />
        ) : (
          <LibraryPage options={options.data} />
        )}
      </main>
    </div>
  );
}

if (!window.location.hash) navigate("/");
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
