// Shapes returned by the Python API (src/sparring/web/app.py).

export type Fighter = "A" | "B";
export type Kind = "strike" | "defense" | "fault" | "other";
export type JobStatus = "queued" | "running" | "done" | "failed";

export interface MoveInfo {
  kind: Kind;
  label: string;
  limb: string;
  side: "lead" | "rear" | null;
}

export interface Options {
  disciplines: { id: string; title: string }[];
  methods: { id: string; help: string }[];
  defaults: { discipline: string; method: string };
  moves: Record<string, MoveInfo>;
  max_upload_mb: number;
}

export interface Progress {
  stage: string;
  stages: string[];
  stage_index: number;
  fraction: number;
}

export interface Job {
  id: string;
  name: string;
  status: JobStatus;
  options: { discipline: string; method: string; fighters: [number, number] | null };
  created: number;
  updated: number;
  error: string | null;
  log_tail: string[];
  duration_s: number | null;
  fps: number | null;
  width: number | null;
  height: number | null;
  progress: Progress | null;
  queue_position: number | null;
  has_thumbnail: boolean;
}

export interface AnalysisEvent {
  fighter: Fighter;
  type: string;
  start_t: number;
  peak_t: number;
  end_t: number;
  confidence: number;
  evidence: Record<string, unknown>;
}

export interface Drill {
  name: string;
  how: string;
  dose: string;
  for?: string;
}

export interface Feedback {
  id: string;
  kind: "strength" | "improve";
  title: string;
  observation: string;
  why: string;
  fix: string;
  drills: Drill[];
  watch: number[];
  score: number;
  basis: string;
  reliability: string;
}

export interface Coaching {
  summary: string;
  strengths: Feedback[];
  priorities: Feedback[];
  session_plan: Drill[];
}

export interface FighterReport {
  identity: string;
  metrics: {
    offense: Record<string, unknown> & { by_type: Record<string, number> };
    defense: Record<string, unknown> & { responses: Record<string, number> };
    guard: Record<string, unknown> & { low_share_by_hand: { lead: number | null; rear: number | null } };
    [k: string]: unknown;
  };
  coaching: Coaching;
}

export interface Report {
  discipline: string;
  detection_method: string;
  data_quality: { side_on_share: number; notes: string[] };
  fighters: Record<Fighter, FighterReport>;
}

export interface Analysis {
  job: Job;
  video: { fps: number; n_frames: number; width: number; height: number };
  report: Report;
  events: AnalysisEvent[];
}

export interface Skeleton {
  keypoints: string[];
  bones: [number, number][];
}
