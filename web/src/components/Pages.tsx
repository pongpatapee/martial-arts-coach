import { useRef, useState } from "react";
import { api } from "../api/client";
import type { Job, Options, Skeleton } from "../api/types";
import { navigate, useInterval, useLoad } from "../hooks/common";
import { formatDuration } from "../lib/time";
import { Viewer } from "./Viewer";

const STAGE_LABEL: Record<string, string> = {
  ingest: "Reading the video",
  track: "Finding people",
  fighters: "Identifying the two fighters",
  pose: "Tracking body pose",
  pose3d: "3D pose",
  kinematics: "Measuring movement",
  events: "Detecting moves",
  report: "Writing the coaching report",
  render: "Rendering",
};

// ---- upload ------------------------------------------------------------------------------------

export function UploadPage({ options }: { options: Options }) {
  const [file, setFile] = useState<File | null>(null);
  const [discipline, setDiscipline] = useState(options.defaults.discipline);
  const [method, setMethod] = useState(options.defaults.method);
  const [fighters, setFighters] = useState("");
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<(() => void) | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const pick = (f: File | undefined) => {
    setError(null);
    if (!f) return;
    if (!f.type.startsWith("video/") && !/\.(mp4|mov|m4v|webm|mkv|avi)$/i.test(f.name)) {
      setError("That doesn't look like a video file.");
      return;
    }
    if (f.size > options.max_upload_mb * 1024 * 1024) {
      setError(`The file is larger than ${options.max_upload_mb} MB.`);
      return;
    }
    setFile(f);
  };

  const submit = async () => {
    if (!file) return;
    setError(null);
    setProgress(0);
    const { promise, abort } = api.upload(
      file,
      { discipline, method, fighters: fighters.trim() || undefined },
      setProgress,
    );
    abortRef.current = abort;
    try {
      const job = await promise;
      navigate(`/jobs/${job.id}`);
    } catch (e) {
      setError((e as Error).message);
      setProgress(null);
    }
  };

  const methodHelp = options.methods.find((m) => m.id === method)?.help;
  return (
    <div className="page narrow">
      <h1>New analysis</h1>
      <div
        className={`dropzone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          pick(e.dataTransfer.files[0]);
        }}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/*,.mov,.mp4,.m4v,.webm,.mkv"
          hidden
          onChange={(e) => pick(e.target.files?.[0])}
        />
        {file ? (
          <>
            <div className="dz-title">{file.name}</div>
            <div className="muted">{(file.size / 1024 / 1024).toFixed(1)} MB · click to choose another</div>
          </>
        ) : (
          <>
            <div className="dz-title">Drop a sparring video here</div>
            <div className="muted">or click to choose · mp4, mov, webm · up to {options.max_upload_mb} MB</div>
          </>
        )}
      </div>

      <div className="form-grid">
        <label>
          <span>Discipline</span>
          <select value={discipline} onChange={(e) => setDiscipline(e.target.value)}>
            {options.disciplines.map((d) => (
              <option key={d.id} value={d.id}>
                {d.title}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Strike detection</span>
          <select value={method} onChange={(e) => setMethod(e.target.value)}>
            {options.methods.map((m) => (
              <option key={m.id} value={m.id}>
                {m.id}
              </option>
            ))}
          </select>
          {methodHelp && <small className="muted">{methodHelp}</small>}
        </label>
        <details className="advanced">
          <summary>Advanced</summary>
          <label>
            <span>Fighter override (track ids)</span>
            <input
              value={fighters}
              onChange={(e) => setFighters(e.target.value)}
              placeholder="e.g. 3,7 – only if the wrong people were picked"
              pattern="\d+,\d+"
            />
          </label>
        </details>
      </div>

      <div className="tips muted small">
        Best results: side-on camera, both fighters in frame, 60 fps if your phone allows. Analysis takes a few minutes
        per minute of video.
      </div>

      {error && <div className="notice error">{error}</div>}
      {progress !== null ? (
        <div className="upload-progress">
          <div className="bar">
            <div style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
          <span>Uploading… {Math.round(progress * 100)}%</span>
          <button className="btn small" onClick={() => abortRef.current?.()}>
            Cancel
          </button>
        </div>
      ) : (
        <button className="btn primary big" onClick={submit} disabled={!file}>
          Analyse video
        </button>
      )}
    </div>
  );
}

// ---- library -----------------------------------------------------------------------------------

export function LibraryPage({ options }: { options: Options }) {
  const jobs = useLoad(api.jobs, "jobs");
  const active = (jobs.data ?? []).some((j) => j.status === "queued" || j.status === "running");
  useInterval(jobs.reload, 3000, active);
  const title = (id: string) => options.disciplines.find((d) => d.id === id)?.title ?? id;

  if (jobs.error) return <div className="page notice error">Could not load analyses: {jobs.error.message}</div>;
  if (!jobs.data) return <div className="page muted">Loading…</div>;
  return (
    <div className="page">
      <div className="page-head">
        <h1>Analyses</h1>
        <button className="btn primary" onClick={() => navigate("/new")}>
          + New analysis
        </button>
      </div>
      {jobs.data.length === 0 ? (
        <div className="empty-state">
          <p>No analyses yet.</p>
          <button className="btn primary" onClick={() => navigate("/new")}>
            Upload your first sparring video
          </button>
        </div>
      ) : (
        <div className="cards">
          {jobs.data.map((j) => (
            <JobCard key={j.id} job={j} discipline={title(j.options.discipline)} onChange={jobs.reload} />
          ))}
        </div>
      )}
    </div>
  );
}

function JobCard({ job, discipline, onChange }: { job: Job; discipline: string; onChange: () => void }) {
  const [busy, setBusy] = useState(false);
  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try {
      await fn();
      onChange();
    } finally {
      setBusy(false);
    }
  };
  return (
    <article className="card">
      <a className="thumb" href={`#/jobs/${job.id}`}>
        {job.has_thumbnail ? <img src={api.thumbnailUrl(job.id)} alt="" loading="lazy" /> : <div className="thumb-empty" />}
        <span className={`status s-${job.status}`}>{job.status}</span>
      </a>
      <div className="card-body">
        <a href={`#/jobs/${job.id}`} className="card-title">
          {job.name}
        </a>
        <div className="muted small">
          {discipline} · {formatDuration(job.duration_s)} · {new Date(job.created * 1000).toLocaleString()}
        </div>
        <div className="card-actions">
          {job.status === "failed" && (
            <button className="btn small" disabled={busy} onClick={() => act(() => api.retry(job.id))}>
              Retry
            </button>
          )}
          {job.status !== "running" && (
            <button
              className="btn small danger"
              disabled={busy}
              onClick={() => {
                if (window.confirm(`Delete “${job.name}” and its analysis?`)) void act(() => api.deleteJob(job.id));
              }}
            >
              Delete
            </button>
          )}
        </div>
      </div>
    </article>
  );
}

// ---- job (progress, then viewer) ---------------------------------------------------------------

export function JobPage({ id, options, skeleton }: { id: string; options: Options; skeleton: Skeleton | null }) {
  const job = useLoad(() => api.job(id), `job:${id}`);
  const pending = !job.data || job.data.status === "queued" || job.data.status === "running";
  useInterval(job.reload, 1500, pending && !job.error);

  if (job.error) return <div className="page notice error">{job.error.message}</div>;
  if (!job.data) return <div className="page muted">Loading…</div>;
  if (job.data.status === "done") return <Done id={id} options={options} skeleton={skeleton} />;
  return <ProgressView job={job.data} onRetry={job.reload} />;
}

function Done({ id, options, skeleton }: { id: string; options: Options; skeleton: Skeleton | null }) {
  const [version, setVersion] = useState(0);
  const analysis = useLoad(() => api.analysis(id), `analysis:${id}:${version}`);
  if (analysis.error) return <div className="page notice error">{analysis.error.message}</div>;
  if (!analysis.data) return <div className="page muted">Loading analysis…</div>;
  return (
    <Viewer
      key={analysis.data.job.updated}
      analysis={analysis.data}
      options={options}
      skeleton={skeleton}
      onReanalyzed={() => setVersion((v) => v + 1)}
    />
  );
}

function ProgressView({ job, onRetry }: { job: Job; onRetry: () => void }) {
  const p = job.progress;
  const [busy, setBusy] = useState(false);
  const stages = p?.stages.filter((s) => s !== "pose3d") ?? [];
  return (
    <div className="page narrow">
      <h1>{job.name}</h1>
      {job.status === "queued" && (
        <p className="muted">Waiting in the queue{job.queue_position ? ` (position ${job.queue_position})` : ""}…</p>
      )}
      {job.status === "running" && !p && <p className="muted">Preparing the video for playback…</p>}
      {job.status === "running" && p && (
        <ol className="stages">
          {stages.map((s) => {
            const i = p.stages.indexOf(s);
            const state = i < p.stage_index ? "done" : i === p.stage_index ? "active" : "todo";
            return (
              <li key={s} className={state}>
                <span className="stage-dot" />
                <span>{STAGE_LABEL[s] ?? s}</span>
                {state === "active" && (
                  <span className="bar">
                    <span style={{ width: `${Math.max(3, Math.round(p.fraction * 100))}%` }} />
                  </span>
                )}
              </li>
            );
          })}
        </ol>
      )}
      {job.status === "failed" && (
        <div className="notice error">
          <b>Analysis failed:</b> {job.error}
          {job.log_tail.length > 0 && <pre className="log">{job.log_tail.join("\n")}</pre>}
          <button
            className="btn"
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              try {
                await api.retry(job.id);
                onRetry();
              } finally {
                setBusy(false);
              }
            }}
          >
            Retry
          </button>
        </div>
      )}
      <p className="muted small">You can leave this page; the analysis continues and appears in the library.</p>
    </div>
  );
}
