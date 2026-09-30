import { decodePoses, parseShape, type Poses } from "../lib/poses";
import type { Analysis, Job, Options, Skeleton } from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const post = (body?: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});

export interface AnalysisOptions {
  discipline: string;
  method: string;
}

export const api = {
  options: () => request<Options>("/api/options"),
  skeleton: () => request<Skeleton>("/api/skeleton"),
  jobs: () => request<Job[]>("/api/jobs"),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  analysis: (id: string) => request<Analysis>(`/api/jobs/${id}/analysis`),
  deleteJob: (id: string) => request<void>(`/api/jobs/${id}`, { method: "DELETE" }),
  retry: (id: string, opts?: AnalysisOptions) => request<Job>(`/api/jobs/${id}/retry`, post(opts)),
  reanalyze: (id: string, opts: AnalysisOptions) => request<Job>(`/api/jobs/${id}/reanalyze`, post(opts)),
  videoUrl: (id: string) => `/api/jobs/${id}/video`,
  thumbnailUrl: (id: string, version = 0) => `/api/jobs/${id}/thumbnail?v=${version}`,

  async poses(id: string): Promise<Poses> {
    const res = await fetch(`/api/jobs/${id}/poses`);
    if (!res.ok) throw new ApiError(res.status, "could not load poses");
    return decodePoses(await res.arrayBuffer(), parseShape(res.headers.get("X-Shape")));
  },

  /** Upload with progress (fetch cannot report upload progress). */
  upload(
    file: File,
    opts: AnalysisOptions & { fighters?: string },
    onProgress: (fraction: number) => void,
  ): { promise: Promise<Job>; abort: () => void } {
    const xhr = new XMLHttpRequest();
    const promise = new Promise<Job>((resolve, reject) => {
      const form = new FormData();
      form.append("file", file);
      form.append("discipline", opts.discipline);
      form.append("method", opts.method);
      if (opts.fighters) form.append("fighters", opts.fighters);
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
      xhr.onload = () => {
        let body: { detail?: string } & Partial<Job> = {};
        try {
          body = JSON.parse(xhr.responseText);
        } catch {
          /* ignore */
        }
        if (xhr.status >= 200 && xhr.status < 300) resolve(body as Job);
        else reject(new ApiError(xhr.status, body.detail ?? `upload failed (${xhr.status})`));
      };
      xhr.onerror = () => reject(new ApiError(0, "network error during upload"));
      xhr.onabort = () => reject(new ApiError(0, "upload cancelled"));
      xhr.open("POST", "/api/jobs");
      xhr.send(form);
    });
    return { promise, abort: () => xhr.abort() };
  },
};
