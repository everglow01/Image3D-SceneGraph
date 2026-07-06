import { ChangeEvent, useMemo, useState } from "react";
import {
  Download,
  FileArchive,
  Image,
  Images,
  Play,
  RefreshCw,
  UploadCloud,
  Video
} from "lucide-react";
import { PointCloudViewer } from "./PointCloudViewer";

type Mode = "image" | "multi_image" | "video" | "panorama";

type Manifest = {
  job_id: string;
  status: string;
  stage: string;
  progress: number;
  mode: Mode;
  input_type: string;
  created_at: string;
  inputs: Array<{
    filename: string;
    path: string;
    content_type: string | null;
    size_bytes: number;
  }>;
  assets: {
    point_cloud?: string;
    scene_graph?: string;
    log?: string;
  };
  metrics: {
    num_inputs: number;
    num_points: number;
    num_objects: number;
  };
};

type SceneGraph = {
  job_id: string;
  mode: Mode;
  coordinate_system: string;
  objects: Array<{
    id: string;
    label: string;
    confidence: number;
    center: number[];
    extent: number[];
    source: string;
  }>;
  relations: unknown[];
  diagnostics: {
    scale_recovered: boolean;
    physical_checks: unknown[];
  };
};

type JobStatus = Pick<
  Manifest,
  "job_id" | "status" | "stage" | "progress" | "mode" | "metrics"
>;

const modeOptions: Array<{
  id: Mode;
  label: string;
  icon: typeof Image;
  fileHint: string;
}> = [
  { id: "image", label: "Image", icon: Image, fileHint: "One image" },
  { id: "multi_image", label: "Multi-image", icon: Images, fileHint: "Two or more images" },
  { id: "video", label: "Video", icon: Video, fileHint: "One video" },
  { id: "panorama", label: "Panorama", icon: FileArchive, fileHint: "One 360 image" }
];

export function App() {
  const [mode, setMode] = useState<Mode>("image");
  const [files, setFiles] = useState<File[]>([]);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [scene, setScene] = useState<SceneGraph | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedMode = modeOptions.find((option) => option.id === mode) ?? modeOptions[0];
  const pointCloudUrl = useMemo(() => {
    if (!manifest?.assets.point_cloud) {
      return null;
    }
    return `/api/jobs/${manifest.job_id}/assets/${manifest.assets.point_cloud}`;
  }, [manifest]);

  function onModeChange(nextMode: Mode) {
    setMode(nextMode);
    setFiles([]);
    setError(null);
  }

  function onFilesChange(event: ChangeEvent<HTMLInputElement>) {
    setFiles(Array.from(event.target.files ?? []));
    setError(null);
  }

  async function createJob() {
    const validationError = validateFiles(mode, files);
    if (validationError) {
      setError(validationError);
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const form = new FormData();
      form.append("mode", mode);
      for (const file of files) {
        form.append("files", file);
      }

      const created = await requestJson<Manifest>("/api/jobs", {
        method: "POST",
        body: form
      });
      setManifest(created);

      const [status, sceneGraph] = await Promise.all([
        requestJson<JobStatus>(`/api/jobs/${created.job_id}`),
        requestJson<SceneGraph>(`/api/jobs/${created.job_id}/scene`)
      ]);

      setJobStatus(status);
      setScene(sceneGraph);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to create job");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function refreshJob() {
    if (!manifest) {
      return;
    }
    setError(null);
    try {
      const [status, nextManifest, sceneGraph] = await Promise.all([
        requestJson<JobStatus>(`/api/jobs/${manifest.job_id}`),
        requestJson<Manifest>(`/api/jobs/${manifest.job_id}/manifest`),
        requestJson<SceneGraph>(`/api/jobs/${manifest.job_id}/scene`)
      ]);
      setJobStatus(status);
      setManifest(nextManifest);
      setScene(sceneGraph);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to refresh job");
    }
  }

  const currentStatus = jobStatus ?? manifest;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Image3D-SceneGraph</p>
          <h1>Scene reconstruction console</h1>
        </div>
        <div className="status-chip">{currentStatus?.status ?? "idle"}</div>
      </header>

      <section className="workspace">
        <aside className="panel upload-panel" aria-label="Job input">
          <div className="panel-heading">
            <h2>Input</h2>
            <span>{selectedMode.fileHint}</span>
          </div>

          <div className="mode-grid" role="group" aria-label="Reconstruction mode">
            {modeOptions.map((option) => {
              const Icon = option.icon;
              return (
                <button
                  className={option.id === mode ? "mode-button active" : "mode-button"}
                  key={option.id}
                  type="button"
                  onClick={() => onModeChange(option.id)}
                >
                  <Icon size={18} aria-hidden="true" />
                  <span>{option.label}</span>
                </button>
              );
            })}
          </div>

          <label className="file-drop">
            <UploadCloud size={22} aria-hidden="true" />
            <span>{files.length > 0 ? `${files.length} selected` : "Choose files"}</span>
            <input
              type="file"
              accept={mode === "video" ? "video/*" : "image/*"}
              multiple={mode === "multi_image"}
              onChange={onFilesChange}
            />
          </label>

          <div className="file-list">
            {files.length === 0 ? (
              <p>No file selected</p>
            ) : (
              files.map((file) => (
                <div className="file-row" key={`${file.name}-${file.size}`}>
                  <span>{file.name}</span>
                  <small>{formatBytes(file.size)}</small>
                </div>
              ))
            )}
          </div>

          {error && <div className="error-box">{error}</div>}

          <button className="primary-button" type="button" onClick={createJob} disabled={isSubmitting}>
            <Play size={18} aria-hidden="true" />
            <span>{isSubmitting ? "Creating job" : "Create job"}</span>
          </button>
        </aside>

        <section className="viewer-column">
          <div className="viewer-header">
            <div>
              <h2>3D viewer</h2>
              <span>{manifest?.assets.point_cloud ?? "No point cloud loaded"}</span>
            </div>
            <button className="icon-button" type="button" onClick={refreshJob} disabled={!manifest}>
              <RefreshCw size={17} aria-hidden="true" />
              <span>Refresh</span>
            </button>
          </div>
          <PointCloudViewer sourceUrl={pointCloudUrl} />
        </section>

        <aside className="panel result-panel" aria-label="Job results">
          <div className="panel-heading">
            <h2>Job</h2>
            <span>{manifest?.job_id ?? "none"}</span>
          </div>

          <dl className="metrics-grid">
            <div>
              <dt>Status</dt>
              <dd>{currentStatus?.status ?? "-"}</dd>
            </div>
            <div>
              <dt>Mode</dt>
              <dd>{currentStatus?.mode ?? "-"}</dd>
            </div>
            <div>
              <dt>Inputs</dt>
              <dd>{currentStatus?.metrics.num_inputs ?? "-"}</dd>
            </div>
            <div>
              <dt>Points</dt>
              <dd>{currentStatus?.metrics.num_points ?? "-"}</dd>
            </div>
          </dl>

          <section className="result-section">
            <h3>Objects</h3>
            {scene?.objects.length ? (
              <div className="object-list">
                {scene.objects.map((object) => (
                  <div className="object-row" key={object.id}>
                    <div>
                      <strong>{object.label}</strong>
                      <span>{object.id}</span>
                    </div>
                    <small>{Math.round(object.confidence * 100)}%</small>
                  </div>
                ))}
              </div>
            ) : (
              <p className="empty-text">No objects</p>
            )}
          </section>

          <section className="result-section">
            <h3>Assets</h3>
            <div className="asset-links">
              <AssetLink manifest={manifest} assetKey="point_cloud" label="Point cloud" />
              <AssetLink manifest={manifest} assetKey="scene_graph" label="Scene graph" />
              <AssetLink manifest={manifest} assetKey="log" label="Run log" />
              {manifest && (
                <a href={`/api/jobs/${manifest.job_id}/download`}>
                  <Download size={16} aria-hidden="true" />
                  <span>Bundle</span>
                </a>
              )}
            </div>
          </section>
        </aside>
      </section>
    </main>
  );
}

function AssetLink({
  manifest,
  assetKey,
  label
}: {
  manifest: Manifest | null;
  assetKey: keyof Manifest["assets"];
  label: string;
}) {
  const asset = manifest?.assets[assetKey];
  if (!manifest || !asset) {
    return (
      <button className="asset-placeholder" type="button" disabled>
        {label}
      </button>
    );
  }
  return (
    <a href={`/api/jobs/${manifest.job_id}/assets/${asset}`}>
      <Download size={16} aria-hidden="true" />
      <span>{label}</span>
    </a>
  );
}

function validateFiles(mode: Mode, files: File[]) {
  if (files.length === 0) {
    return "Select at least one file.";
  }
  if ((mode === "image" || mode === "video" || mode === "panorama") && files.length !== 1) {
    return `${mode} mode requires exactly one file.`;
  }
  if (mode === "multi_image" && files.length < 2) {
    return "Multi-image mode requires at least two files.";
  }
  return null;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        message = payload.detail;
      }
    } catch {
      // Keep the HTTP status message when the response is not JSON.
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

function formatBytes(bytes: number) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
