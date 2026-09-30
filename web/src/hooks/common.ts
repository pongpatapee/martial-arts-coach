import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";

// ---- hash routing ------------------------------------------------------------------------------

export type Route = { page: "library" } | { page: "new" } | { page: "job"; id: string };

export function parseRoute(hash: string): Route {
  const path = hash.replace(/^#/, "") || "/";
  const job = path.match(/^\/jobs\/([a-f0-9]+)$/);
  if (job) return { page: "job", id: job[1]! };
  if (path === "/new") return { page: "new" };
  return { page: "library" };
}

const subscribeHash = (cb: () => void) => {
  window.addEventListener("hashchange", cb);
  return () => window.removeEventListener("hashchange", cb);
};

export function useRoute(): Route {
  const hash = useSyncExternalStore(subscribeHash, () => window.location.hash);
  return parseRoute(hash);
}

export function navigate(path: string): void {
  window.location.hash = path;
}

// ---- data loading ------------------------------------------------------------------------------

export interface Loadable<T> {
  data: T | null;
  error: Error | null;
  loading: boolean;
  reload: () => void;
}

/** Minimal async loader: re-runs when `key` changes, ignores stale responses. */
export function useLoad<T>(fn: () => Promise<T>, key: string): Loadable<T> {
  const [state, setState] = useState<{ data: T | null; error: Error | null; loading: boolean }>({
    data: null,
    error: null,
    loading: true,
  });
  const [nonce, setNonce] = useState(0);
  const fnRef = useRef(fn);
  fnRef.current = fn;
  useEffect(() => {
    let live = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    fnRef
      .current()
      .then((data) => live && setState({ data, error: null, loading: false }))
      .catch((error: Error) => live && setState((s) => ({ ...s, error, loading: false })));
    return () => {
      live = false;
    };
  }, [key, nonce]);
  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { ...state, reload };
}

/** Call `fn` every `ms` while `active`. */
export function useInterval(fn: () => void, ms: number, active: boolean): void {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => ref.current(), ms);
    return () => window.clearInterval(id);
  }, [ms, active]);
}

// ---- keyboard ----------------------------------------------------------------------------------

/** Global shortcuts, ignored while typing in form fields. Handlers keyed by KeyboardEvent.key;
 * prefix with "shift+" for shifted variants. Return false from a handler to let it through. */
export function useShortcuts(handlers: Record<string, (e: KeyboardEvent) => void | boolean>): void {
  const ref = useRef(handlers);
  ref.current = handlers;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName))) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const key = (e.shiftKey && e.key.length > 1 ? "shift+" : "") + e.key;
      const h = ref.current[key] ?? ref.current[e.key];
      if (h && h(e) !== false) e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}

// ---- theme -------------------------------------------------------------------------------------

export type Theme = "light" | "dark" | "system";

export function useTheme(): [Theme, (t: Theme) => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      return (localStorage.getItem("theme") as Theme) || "system";
    } catch {
      return "system";
    }
  });
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("theme", theme);
    } catch {
      /* private mode */
    }
  }, [theme]);
  return [theme, setTheme];
}
