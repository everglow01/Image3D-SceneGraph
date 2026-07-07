import { ChangeEvent, useEffect, useMemo, useState } from "react";
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
import { GeometryViewer } from "./GeometryViewer";

type Mode = "image" | "multi_image" | "video" | "panorama" | "imported_asset";
type GeometryBackend = "mock" | "vggt" | "dust3r" | "mast3r" | "nerfstudio_3dgs";
type OutputType = "point_cloud" | "mesh" | "gaussian_splat";

type Manifest = {
  job_id: string;
  status: string;
  stage: string;
  progress: number;
  mode: Mode;
  input_type: string;
  geometry_backend: GeometryBackend;
  output_type: OutputType;
  created_at: string;
  inputs: Array<{
    filename: string;
    path: string;
    content_type: string | null;
    size_bytes: number;
  }>;
  assets: {
    point_cloud?: string;
    mesh?: string;
    scene_splat?: string;
    scene_graph?: string;
    log?: string;
  };
  metrics: {
    num_inputs: number;
    num_points: number;
    num_objects: number;
    num_groups?: number;
    batch_size?: number;
    overlap_size?: number;
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
  | "job_id"
  | "status"
  | "stage"
  | "progress"
  | "mode"
  | "geometry_backend"
  | "output_type"
  | "metrics"
>;

type BackendStatus = {
  id: GeometryBackend;
  label: string;
  available: boolean;
  reason: string | null;
  supported_outputs: OutputType[];
  setup_command: string | null;
};

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

const backendOptions: Array<{
  id: GeometryBackend;
  label: string;
}> = [
  { id: "mock", label: "Mock" },
  { id: "vggt", label: "VGGT" },
  { id: "dust3r", label: "DUSt3R" },
  { id: "mast3r", label: "MASt3R" },
  { id: "nerfstudio_3dgs", label: "Nerfstudio 3DGS" }
];

const outputOptions: Array<{
  id: OutputType;
  label: string;
}> = [
  { id: "point_cloud", label: "Point cloud" },
  { id: "mesh", label: "Mesh" },
  { id: "gaussian_splat", label: "Gaussian splat" }
];

export function App() {
  const [mode, setMode] = useState<Mode>("image");
  const [geometryBackend, setGeometryBackend] = useState<GeometryBackend>("mock");
  const [outputType, setOutputType] = useState<OutputType>("point_cloud");
  const [vggtMaxImages, setVggtMaxImages] = useState(225);
  const [vggtBatchSize, setVggtBatchSize] = useState(8);
  const [vggtOverlapSize, setVggtOverlapSize] = useState(4);
  const [files, setFiles] = useState<File[]>([]);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [scene, setScene] = useState<SceneGraph | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoadingJob, setIsLoadingJob] = useState(false);
  const [jobIdInput, setJobIdInput] = useState("");
  const [backendStatuses, setBackendStatuses] = useState<Record<GeometryBackend, BackendStatus> | null>(null);
  const [error, setError] = useState<string | null>(null);

  const selectedMode = modeOptions.find((option) => option.id === mode) ?? modeOptions[0];
  const selectedBackendStatus = backendStatuses?.[geometryBackend];
  const selectedBackendAvailable = selectedBackendStatus?.available ?? geometryBackend === "mock";
  const selectedOutputSupported =
    selectedBackendStatus?.supported_outputs.includes(outputType) ?? outputType === "point_cloud";
  const pointCloudUrl = useMemo(() => {
    if (!manifest?.assets.point_cloud) {
      return null;
    }
    return `/api/jobs/${manifest.job_id}/assets/${manifest.assets.point_cloud}`;
  }, [manifest]);
  const splatUrl = useMemo(() => {
    if (!manifest?.assets.scene_splat) {
      return null;
    }
    return `/api/jobs/${manifest.job_id}/assets/${manifest.assets.scene_splat}`;
  }, [manifest]);

  useEffect(() => {
    let cancelled = false;

    requestJson<{ backends: BackendStatus[] }>("/api/backends")
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setBackendStatuses(
          Object.fromEntries(payload.backends.map((backend) => [backend.id, backend])) as Record<
            GeometryBackend,
            BackendStatus
          >
        );
      })
      .catch(() => {
        if (!cancelled) {
          setBackendStatuses(null);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

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
    if (!selectedBackendAvailable) {
      setError(selectedBackendStatus?.reason ?? "Selected geometry backend is not available.");
      return;
    }
    if (!selectedOutputSupported) {
      setError(`${outputType} is not supported by ${geometryBackend}.`);
      return;
    }

    const validationError = validateFiles(mode, files);
    if (validationError) {
      setError(validationError);
      return;
    }
    if (geometryBackend === "vggt") {
      const optionError = validateVggtOptions(vggtMaxImages, vggtBatchSize, vggtOverlapSize, files.length);
      if (optionError) {
        setError(optionError);
        return;
      }
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const form = new FormData();
      form.append("mode", mode);
      form.append("geometry_backend", geometryBackend);
      form.append("output_type", outputType);
      if (geometryBackend === "vggt") {
        form.append("vggt_max_images", String(vggtMaxImages));
        form.append("vggt_batch_size", String(vggtBatchSize));
        form.append("vggt_overlap_size", String(vggtOverlapSize));
      }
      for (const file of files) {
        form.append("files", file, getUploadName(file));
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

  async function loadJobById() {
    const jobId = jobIdInput.trim();
    if (!jobId) {
      setError("Enter a job id.");
      return;
    }

    setIsLoadingJob(true);
    setError(null);
    try {
      const [status, nextManifest, sceneGraph] = await Promise.all([
        requestJson<JobStatus>(`/api/jobs/${jobId}`),
        requestJson<Manifest>(`/api/jobs/${jobId}/manifest`),
        requestJson<SceneGraph>(`/api/jobs/${jobId}/scene`)
      ]);
      setJobStatus(status);
      setManifest(nextManifest);
      setScene(sceneGraph);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to load job");
    } finally {
      setIsLoadingJob(false);
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

          <div className="control-stack">
            <label>
              <span>Geometry backend</span>
              <select
                value={geometryBackend}
                onChange={(event) => setGeometryBackend(event.target.value as GeometryBackend)}
              >
                {backendOptions.map((option) => (
                  <option disabled={!isBackendAvailable(option.id, backendStatuses)} key={option.id} value={option.id}>
                    {formatBackendOption(option, backendStatuses)}
                  </option>
                ))}
              </select>
              {selectedBackendStatus?.reason && <small>{selectedBackendStatus.reason}</small>}
              {selectedBackendStatus?.setup_command && <small>{selectedBackendStatus.setup_command}</small>}
            </label>

            <label>
              <span>Output type</span>
              <select
                value={outputType}
                onChange={(event) => setOutputType(event.target.value as OutputType)}
              >
                {outputOptions.map((option) => (
                  <option disabled={!isOutputSupported(option.id, selectedBackendStatus)} key={option.id} value={option.id}>
                    {isOutputSupported(option.id, selectedBackendStatus) ? option.label : `${option.label} (unavailable)`}
                  </option>
                ))}
              </select>
            </label>

            {geometryBackend === "vggt" && (
              <div className="numeric-grid">
                <label>
                  <span>Max images</span>
                  <input
                    type="number"
                    min={1}
                    max={500}
                    step={1}
                    value={vggtMaxImages}
                    onChange={(event) => setVggtMaxImages(Number(event.target.value))}
                  />
                </label>
                <label>
                  <span>Batch size</span>
                  <input
                    type="number"
                    min={1}
                    max={8}
                    step={1}
                    value={vggtBatchSize}
                    onChange={(event) => setVggtBatchSize(Number(event.target.value))}
                  />
                </label>
                <label>
                  <span>Overlap</span>
                  <input
                    type="number"
                    min={1}
                    max={7}
                    step={1}
                    value={vggtOverlapSize}
                    onChange={(event) => setVggtOverlapSize(Number(event.target.value))}
                  />
                </label>
              </div>
            )}
          </div>

          <div className="file-picker-grid">
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

            {mode === "multi_image" && (
              <label className="file-drop compact">
                <UploadCloud size={20} aria-hidden="true" />
                <span>Choose folder</span>
                <input
                  {...folderInputProps}
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={onFilesChange}
                />
              </label>
            )}
          </div>

          <div className="file-list">
            {files.length === 0 ? (
              <p>No file selected</p>
            ) : (
              files.map((file) => (
                <div className="file-row" key={`${getUploadName(file)}-${file.size}`}>
                  <span title={getUploadName(file)}>{getUploadName(file)}</span>
                  <small>{formatBytes(file.size)}</small>
                </div>
              ))
            )}
          </div>

          {error && <div className="error-box">{error}</div>}

          <button
            className="primary-button"
            type="button"
            onClick={createJob}
            disabled={isSubmitting || !selectedBackendAvailable || !selectedOutputSupported}
          >
            <Play size={18} aria-hidden="true" />
            <span>{isSubmitting ? "Creating job" : "Create job"}</span>
          </button>
        </aside>

        <section className="viewer-column">
          <div className="viewer-header">
            <div>
              <h2>3D viewer</h2>
              <span>{manifest?.assets.scene_splat ?? manifest?.assets.point_cloud ?? "No geometry loaded"}</span>
            </div>
            <button className="icon-button" type="button" onClick={refreshJob} disabled={!manifest}>
              <RefreshCw size={17} aria-hidden="true" />
              <span>Refresh</span>
            </button>
          </div>
          <GeometryViewer pointCloudUrl={pointCloudUrl} splatUrl={splatUrl} />
        </section>

        <aside className="panel result-panel" aria-label="Job results">
          <div className="panel-heading">
            <h2>Job</h2>
            <span>{manifest?.job_id ?? "none"}</span>
          </div>

          <div className="load-job-row">
            <input
              aria-label="Job id"
              placeholder="Load job id"
              value={jobIdInput}
              onChange={(event) => setJobIdInput(event.target.value)}
            />
            <button type="button" onClick={loadJobById} disabled={isLoadingJob}>
              {isLoadingJob ? "Loading" : "Load"}
            </button>
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
              <dt>Backend</dt>
              <dd>{currentStatus?.geometry_backend ?? "-"}</dd>
            </div>
            <div>
              <dt>Output</dt>
              <dd>{currentStatus?.output_type ?? "-"}</dd>
            </div>
            <div>
              <dt>Inputs</dt>
              <dd>{currentStatus?.metrics.num_inputs ?? "-"}</dd>
            </div>
            <div>
              <dt>Points</dt>
              <dd>{currentStatus?.metrics.num_points ?? "-"}</dd>
            </div>
            <div>
              <dt>Groups</dt>
              <dd>{currentStatus?.metrics.num_groups ?? "-"}</dd>
            </div>
            <div>
              <dt>Batch</dt>
              <dd>{currentStatus?.metrics.batch_size ?? "-"}</dd>
            </div>
            <div>
              <dt>Overlap</dt>
              <dd>{currentStatus?.metrics.overlap_size ?? "-"}</dd>
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
              <AssetLink manifest={manifest} assetKey="mesh" label="Mesh" />
              <AssetLink manifest={manifest} assetKey="scene_splat" label="Gaussian splat" />
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

const folderInputProps: Record<string, string> = {
  webkitdirectory: "",
  directory: ""
};

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

function validateVggtOptions(maxImages: number, batchSize: number, overlapSize: number, fileCount: number) {
  if (!Number.isInteger(maxImages) || maxImages <= 0) {
    return "VGGT max images must be a positive integer.";
  }
  if (!Number.isInteger(batchSize) || batchSize <= 0) {
    return "VGGT batch size must be a positive integer.";
  }
  if (!Number.isInteger(overlapSize) || overlapSize <= 0) {
    return "VGGT overlap must be a positive integer.";
  }
  if (fileCount > 1 && batchSize < 2) {
    return "VGGT batch size must be at least 2 for multi-image jobs.";
  }
  if (overlapSize >= batchSize) {
    return "VGGT overlap must be smaller than batch size.";
  }
  if (batchSize > maxImages) {
    return "VGGT batch size cannot be larger than max images.";
  }
  return null;
}

function isBackendAvailable(
  backendId: GeometryBackend,
  statuses: Record<GeometryBackend, BackendStatus> | null
) {
  return statuses?.[backendId]?.available ?? backendId === "mock";
}

function formatBackendOption(
  option: { id: GeometryBackend; label: string },
  statuses: Record<GeometryBackend, BackendStatus> | null
) {
  const status = statuses?.[option.id];
  if (!status) {
    return option.id === "mock" ? option.label : `${option.label} (checking)`;
  }
  return status.available ? status.label : `${status.label} (setup required)`;
}

function isOutputSupported(outputType: OutputType, backendStatus: BackendStatus | undefined) {
  return backendStatus?.supported_outputs.includes(outputType) ?? outputType === "point_cloud";
}

function getUploadName(file: File) {
  return (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;
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
