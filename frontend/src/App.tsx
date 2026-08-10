import { ChangeEvent, useEffect, useMemo, useState } from "react";
import {
  Download,
  FileArchive,
  Image,
  Images,
  Play,
  RefreshCw,
  RotateCcw,
  Square,
  UploadCloud,
  Video
} from "lucide-react";
import { GeometryViewer } from "./GeometryViewer";
import {
  findGaussianTrainerStatus,
  formatGaussianTrainerOption
} from "./trainerOptions";
import type { GaussianTrainer, GaussianTrainerStatus } from "./trainerOptions";

type Mode = "image" | "multi_image" | "video" | "panorama" | "imported_asset";
type GeometryBackend = "mock" | "vggt" | "colmap" | "colmap_vggt" | "dust3r" | "mast3r" | "project_3dgs";
type OutputType = "point_cloud" | "mesh" | "gaussian_splat";
type MeshMethod = "poisson" | "ball_pivoting" | "alpha_shape";
type ViewerMode = "point_cloud" | "mesh" | "gaussian_splat";
type ConfidenceThresholdScope = "global" | "per_frame";
type ConsistencySupportPolicy = "any_support" | "adaptive_two";
type PointBudgetPolicy = "random" | "spatial_balanced";
type ColmapVggtGrouping = "sequential" | "covisibility";

type MeshSettings = {
  method: MeshMethod;
  voxel_size: number;
  normal_radius: number;
  statistical_std_ratio: number;
  poisson_depth: number;
  density_trim_quantile: number;
  component_min_ratio: number;
  edge_trim_factor: number;
  max_triangles: number;
  alpha: number;
};

type MeshVariant = {
  id: string;
  label: string;
  method: MeshMethod;
  mesh_asset: string;
  diagnostics_asset: string;
  source_asset: string;
  options: Partial<MeshSettings>;
  metrics: Record<string, string | number | boolean>;
  created_at: string;
};

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
  updated_at?: string;
  started_at?: string | null;
  completed_at?: string | null;
  active_attempt_id?: string | null;
  error?: { code: string; message: string } | null;
  inputs: Array<{
    filename: string;
    path: string;
    content_type: string | null;
    size_bytes: number;
  }>;
  assets: {
    point_cloud?: string;
    point_cloud_aligned?: string;
    cameras?: string;
    alignment_diagnostics?: string;
    fusion_diagnostics?: string;
    visibility_graph?: string;
    consistency_diagnostics?: string;
    mesh?: string;
    mesh_diagnostics?: string;
    scene_splat?: string;
    gaussian_model?: string;
    gaussian_training_result?: string;
    gaussian_progress?: string;
    gaussian_dataset?: string;
    gaussian_evaluation?: string;
    gaussian_test_evaluation?: string;
    gaussian_test_decision?: string;
    gaussian_export_metadata?: string;
    gaussian_canonical?: string;
    gaussian_camera_path?: string;
    gaussian_bundle?: string;
    scene_graph?: string;
    log?: string;
  };
  gaussian_trainer?: {
    id: GaussianTrainer;
    label: string;
    revision: string;
    license: string;
  };
  mesh_variants?: MeshVariant[];
  metrics: {
    num_inputs: number;
    num_points: number;
    num_objects: number;
    num_groups?: number;
    batch_size?: number;
    vggt_batch_size?: number;
    vggt_grouping?: ColmapVggtGrouping;
    vggt_overlap_size?: number;
    overlap_size?: number;
    alignment_status?: string;
    alignment_plane_inlier_ratio?: number;
    consistency_acceptance_rate?: number;
    consistency_rejected?: number;
    consistency_residual_p90?: number;
    confidence_threshold_scope?: ConfidenceThresholdScope;
    consistency_support_policy?: ConsistencySupportPolicy;
    point_budget_policy?: PointBudgetPolicy;
    point_budget_input_points?: number;
    point_budget_output_points?: number;
    point_budget_applied?: boolean;
    mesh_status?: string;
    mesh_vertices?: number;
    mesh_triangles?: number;
    mesh_processed_points?: number;
    mesh_method?: string;
    mesh_component_count?: number;
    mesh_long_edge_removed_triangles?: number;
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
  | "active_attempt_id"
  | "created_at"
  | "updated_at"
  | "started_at"
  | "completed_at"
  | "error"
  | "metrics"
>;

type BackendStatus = {
  id: GeometryBackend;
  label: string;
  available: boolean;
  reason: string | null;
  supported_outputs: OutputType[];
  setup_command: string | null;
  gaussian_trainers?: GaussianTrainerStatus[];
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
  { id: "colmap", label: "COLMAP" },
  { id: "colmap_vggt", label: "COLMAP + VGGT" },
  { id: "dust3r", label: "DUSt3R" },
  { id: "mast3r", label: "MASt3R" },
  { id: "project_3dgs", label: "Project 3DGS" }
];

const outputOptions: Array<{
  id: OutputType;
  label: string;
}> = [
  { id: "point_cloud", label: "Point cloud" },
  { id: "mesh", label: "Mesh" },
  { id: "gaussian_splat", label: "Gaussian splat" }
];

const defaultMeshSettings: MeshSettings = {
  method: "poisson",
  voxel_size: 0.05,
  normal_radius: 0.2,
  statistical_std_ratio: 2.0,
  poisson_depth: 8,
  density_trim_quantile: 0.1,
  component_min_ratio: 0.03,
  edge_trim_factor: 2.5,
  max_triangles: 120_000,
  alpha: 0.12
};

export function App() {
  const [mode, setMode] = useState<Mode>("image");
  const [geometryBackend, setGeometryBackend] = useState<GeometryBackend>("mock");
  const [outputType, setOutputType] = useState<OutputType>("point_cloud");
  const [gaussianTrainer, setGaussianTrainer] = useState<GaussianTrainer>("graphdeco");
  const [vggtMaxImages, setVggtMaxImages] = useState(225);
  const [vggtBatchSize, setVggtBatchSize] = useState(8);
  const [vggtOverlapSize, setVggtOverlapSize] = useState(4);
  const [colmapVggtBatchSize, setColmapVggtBatchSize] = useState(4);
  const [colmapVggtGrouping, setColmapVggtGrouping] =
    useState<ColmapVggtGrouping>("sequential");
  const [colmapVggtOverlapSize, setColmapVggtOverlapSize] = useState(2);
  const [colmapVggtMaxPoints, setColmapVggtMaxPoints] = useState(2_000_000);
  const [colmapVggtConfPercentile, setColmapVggtConfPercentile] = useState(50);
  const [confidenceThresholdScope, setConfidenceThresholdScope] =
    useState<ConfidenceThresholdScope>("global");
  const [consistencySupportPolicy, setConsistencySupportPolicy] =
    useState<ConsistencySupportPolicy>("any_support");
  const [pointBudgetPolicy, setPointBudgetPolicy] = useState<PointBudgetPolicy>("random");
  const [files, setFiles] = useState<File[]>([]);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [scene, setScene] = useState<SceneGraph | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoadingJob, setIsLoadingJob] = useState(false);
  const [jobIdInput, setJobIdInput] = useState("");
  const [pointCloudVariant, setPointCloudVariant] = useState<"raw" | "aligned">("aligned");
  const [viewerMode, setViewerMode] = useState<ViewerMode>("point_cloud");
  const [meshSettings, setMeshSettings] = useState<MeshSettings>(defaultMeshSettings);
  const [selectedMeshVariantId, setSelectedMeshVariantId] = useState<string | null>(null);
  const [isBuildingMesh, setIsBuildingMesh] = useState(false);
  const [isChangingLifecycle, setIsChangingLifecycle] = useState(false);
  const [backendStatuses, setBackendStatuses] = useState<Record<GeometryBackend, BackendStatus> | null>(null);
  const [error, setError] = useState<string | null>(null);

  const selectedMode = modeOptions.find((option) => option.id === mode) ?? modeOptions[0];
  const selectedBackendStatus = backendStatuses?.[geometryBackend];
  const gaussianTrainerStatuses = selectedBackendStatus?.gaussian_trainers ?? [];
  const selectedGaussianTrainerStatus = findGaussianTrainerStatus(
    gaussianTrainerStatuses,
    gaussianTrainer
  );
  const selectedBackendAvailable = selectedBackendStatus?.available ?? geometryBackend === "mock";
  const selectedOutputSupported =
    selectedBackendStatus?.supported_outputs.includes(outputType) ?? outputType === "point_cloud";
  const selectedPointCloudAsset =
    pointCloudVariant === "aligned" && manifest?.assets.point_cloud_aligned
      ? manifest.assets.point_cloud_aligned
      : manifest?.assets.point_cloud;
  const pointCloudUrl = useMemo(() => {
    if (!manifest || !selectedPointCloudAsset) {
      return null;
    }
    return `/api/jobs/${manifest.job_id}/assets/${selectedPointCloudAsset}`;
  }, [manifest, selectedPointCloudAsset]);
  const camerasUrl = useMemo(() => {
    if (!manifest?.assets.cameras) {
      return null;
    }
    return `/api/jobs/${manifest.job_id}/assets/${manifest.assets.cameras}`;
  }, [manifest]);
  const alignmentDiagnosticsUrl = useMemo(() => {
    if (!manifest?.assets.alignment_diagnostics) {
      return null;
    }
    return `/api/jobs/${manifest.job_id}/assets/${manifest.assets.alignment_diagnostics}`;
  }, [manifest]);
  const splatUrl = useMemo(() => {
    if (!manifest?.assets.scene_splat) {
      return null;
    }
    return `/api/jobs/${manifest.job_id}/assets/${manifest.assets.scene_splat}`;
  }, [manifest]);
  const splatMetadataUrl = useMemo(() => {
    if (!manifest?.assets.gaussian_export_metadata) {
      return null;
    }
    return `/api/jobs/${manifest.job_id}/assets/${manifest.assets.gaussian_export_metadata}`;
  }, [manifest]);
  const splatCameraPathUrl = useMemo(() => {
    if (!manifest?.assets.gaussian_camera_path) {
      return null;
    }
    return `/api/jobs/${manifest.job_id}/assets/${manifest.assets.gaussian_camera_path}`;
  }, [manifest]);
  const meshVariants = manifest?.mesh_variants ?? [];
  const selectedMeshVariant =
    meshVariants.find((variant) => variant.id === selectedMeshVariantId) ?? meshVariants[0] ?? null;
  const selectedMeshAsset = selectedMeshVariant?.mesh_asset ?? manifest?.assets.mesh;
  const meshUrl = useMemo(() => {
    if (!manifest || !selectedMeshAsset) {
      return null;
    }
    const cacheKey = selectedMeshVariant?.id ?? "primary";
    return `/api/jobs/${manifest.job_id}/assets/${selectedMeshAsset}?mesh_variant=${encodeURIComponent(cacheKey)}`;
  }, [manifest, selectedMeshAsset, selectedMeshVariant]);
  const hasPointCloud = Boolean(manifest?.assets.point_cloud || manifest?.assets.point_cloud_aligned);
  const hasMesh = Boolean(selectedMeshAsset);
  const hasSplat = Boolean(manifest?.assets.scene_splat);
  const viewerAsset =
    viewerMode === "point_cloud"
      ? selectedPointCloudAsset
      : viewerMode === "mesh"
        ? selectedMeshAsset
        : manifest?.assets.scene_splat;
  const visiblePointCloudUrl = viewerMode === "point_cloud" ? pointCloudUrl : null;
  const visibleMeshUrl = viewerMode === "mesh" ? meshUrl : null;
  const visibleSplatUrl = viewerMode === "gaussian_splat" ? splatUrl : null;

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

  useEffect(() => {
    if (!manifest || !["queued", "running", "exporting"].includes(manifest.status)) {
      return;
    }
    let cancelled = false;
    const interval = window.setInterval(async () => {
      try {
        const nextManifest = await requestJson<Manifest>(`/api/jobs/${manifest.job_id}/manifest`);
        if (cancelled) {
          return;
        }
        applyManifest(nextManifest);
        setJobStatus(nextManifest);
        if (nextManifest.status === "done") {
          setScene(await requestJson<SceneGraph>(`/api/jobs/${manifest.job_id}/scene`));
        }
      } catch (caught) {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : "Failed to refresh job");
        }
      }
    }, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [manifest?.job_id, manifest?.status]);

  function applyManifest(nextManifest: Manifest, selectNewestMeshVariant = false, preferPointCloud = false) {
    setManifest(nextManifest);
    setPointCloudVariant(nextManifest.assets.point_cloud_aligned ? "aligned" : "raw");
    setViewerMode((current) => {
      const hasPointCloud = Boolean(nextManifest.assets.point_cloud || nextManifest.assets.point_cloud_aligned);
      const hasMesh = Boolean(nextManifest.assets.mesh);
      const hasSplat = Boolean(nextManifest.assets.scene_splat);
      if (preferPointCloud && hasPointCloud) {
        return "point_cloud";
      }
      if (
        (current === "point_cloud" && hasPointCloud) ||
        (current === "mesh" && hasMesh) ||
        (current === "gaussian_splat" && hasSplat)
      ) {
        return current;
      }
      if (hasPointCloud) {
        return "point_cloud";
      }
      if (hasMesh) {
        return "mesh";
      }
      return "gaussian_splat";
    });
    const variants = nextManifest.mesh_variants ?? [];
    setSelectedMeshVariantId((current) => {
      if (selectNewestMeshVariant && variants.length > 0) {
        return variants[variants.length - 1].id;
      }
      if (current && variants.some((variant) => variant.id === current)) {
        return current;
      }
      return variants[0]?.id ?? null;
    });
  }

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
    if (geometryBackend === "project_3dgs" && selectedGaussianTrainerStatus?.available === false) {
      setError(selectedGaussianTrainerStatus.reason ?? "Selected Gaussian trainer is unavailable.");
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
    if (geometryBackend === "colmap_vggt") {
      const optionError = validateColmapVggtOptions(
        colmapVggtBatchSize,
        colmapVggtOverlapSize,
        colmapVggtMaxPoints,
        colmapVggtConfPercentile
      );
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
      if (geometryBackend === "project_3dgs") {
        form.append("gaussian_trainer", gaussianTrainer);
      }
      if (geometryBackend === "vggt") {
        form.append("vggt_max_images", String(vggtMaxImages));
        form.append("vggt_batch_size", String(vggtBatchSize));
        form.append("vggt_overlap_size", String(vggtOverlapSize));
      }
      if (geometryBackend === "colmap_vggt") {
        form.append("vggt_batch_size", String(colmapVggtBatchSize));
        form.append("colmap_vggt_grouping", colmapVggtGrouping);
        form.append("colmap_vggt_overlap_size", String(colmapVggtOverlapSize));
        form.append("colmap_vggt_max_points", String(colmapVggtMaxPoints));
        form.append("colmap_vggt_conf_percentile", String(colmapVggtConfPercentile));
        form.append("colmap_vggt_confidence_threshold_scope", confidenceThresholdScope);
        form.append("colmap_vggt_consistency_support_policy", consistencySupportPolicy);
        form.append("colmap_vggt_point_budget_policy", pointBudgetPolicy);
      }
      for (const file of files) {
        form.append("files", file, getUploadName(file));
      }

      const created = await requestJson<Manifest>("/api/jobs", {
        method: "POST",
        body: form
      });
      applyManifest(created, false, true);
      setJobStatus(created);
      setScene(null);
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
      const [status, nextManifest] = await Promise.all([
        requestJson<JobStatus>(`/api/jobs/${manifest.job_id}`),
        requestJson<Manifest>(`/api/jobs/${manifest.job_id}/manifest`)
      ]);
      setJobStatus(status);
      applyManifest(nextManifest);
      if (nextManifest.status === "done") {
        setScene(await requestJson<SceneGraph>(`/api/jobs/${manifest.job_id}/scene`));
      }
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
      const [status, nextManifest] = await Promise.all([
        requestJson<JobStatus>(`/api/jobs/${jobId}`),
        requestJson<Manifest>(`/api/jobs/${jobId}/manifest`)
      ]);
      setJobStatus(status);
      applyManifest(nextManifest, false, true);
      setScene(
        nextManifest.status === "done"
          ? await requestJson<SceneGraph>(`/api/jobs/${jobId}/scene`)
          : null
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to load job");
    } finally {
      setIsLoadingJob(false);
    }
  }

  async function changeLifecycle(action: "cancel" | "retry") {
    if (!manifest) {
      return;
    }
    setIsChangingLifecycle(true);
    setError(null);
    try {
      const nextManifest = await requestJson<Manifest>(
        `/api/jobs/${manifest.job_id}/${action}`,
        { method: "POST" }
      );
      applyManifest(nextManifest);
      setJobStatus(nextManifest);
      if (action === "retry") {
        setScene(null);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : `Failed to ${action} job`);
    } finally {
      setIsChangingLifecycle(false);
    }
  }

  function updateMeshSetting(key: Exclude<keyof MeshSettings, "method">, value: number) {
    setMeshSettings((current) => ({ ...current, [key]: value }));
  }

  async function buildMeshVariant() {
    if (!manifest) {
      return;
    }
    const validationError = validateMeshSettings(meshSettings);
    if (validationError) {
      setError(validationError);
      return;
    }

    setIsBuildingMesh(true);
    setError(null);
    try {
      const nextManifest = await requestJson<Manifest>(`/api/jobs/${manifest.job_id}/mesh-variants`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(meshSettings)
      });
      applyManifest(nextManifest, true);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to build mesh variant");
    } finally {
      setIsBuildingMesh(false);
    }
  }

  const currentStatus = jobStatus ?? manifest;
  const canCancel = Boolean(currentStatus && ["queued", "running", "exporting"].includes(currentStatus.status));
  const canRetry = Boolean(currentStatus && ["failed", "cancelled"].includes(currentStatus.status));
  const hasAlignedPointCloud = Boolean(manifest?.assets.point_cloud_aligned);
  const canBuildMeshVariant = Boolean(manifest?.assets.point_cloud || manifest?.assets.point_cloud_aligned);

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

            {geometryBackend === "project_3dgs" && (
              <label>
                <span>Trainer</span>
                <select
                  value={gaussianTrainer}
                  onChange={(event) => setGaussianTrainer(event.target.value as GaussianTrainer)}
                >
                  {(gaussianTrainerStatuses.length > 0
                    ? gaussianTrainerStatuses
                    : [
                        {
                          id: "graphdeco" as const,
                          label: "Graphdeco official",
                          available: true,
                          reason: null,
                          setup_command: null,
                          revision: "unknown",
                          license: "Graphdeco research/evaluation only"
                        },
                        {
                          id: "project" as const,
                          label: "Project (gsplat)",
                          available: true,
                          reason: null,
                          setup_command: null,
                          revision: "unknown",
                          license: "Apache-2.0"
                        }
                      ]
                  ).map((trainer) => (
                    <option disabled={!trainer.available} key={trainer.id} value={trainer.id}>
                      {formatGaussianTrainerOption(trainer)}
                    </option>
                  ))}
                </select>
                {selectedGaussianTrainerStatus?.reason && (
                  <small>{selectedGaussianTrainerStatus.reason}</small>
                )}
                {selectedGaussianTrainerStatus?.setup_command && (
                  <small>{selectedGaussianTrainerStatus.setup_command}</small>
                )}
              </label>
            )}

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

            {geometryBackend === "colmap_vggt" && (
              <>
                <div className="numeric-grid">
                  <label>
                    <span>Max points</span>
                    <input
                      type="number"
                      min={100000}
                      max={10000000}
                      step={100000}
                      value={colmapVggtMaxPoints}
                      onChange={(event) => setColmapVggtMaxPoints(Number(event.target.value))}
                    />
                  </label>
                  <label>
                    <span>Depth batch</span>
                    <input
                      type="number"
                      min={2}
                      max={8}
                      step={1}
                      value={colmapVggtBatchSize}
                      onChange={(event) => setColmapVggtBatchSize(Number(event.target.value))}
                    />
                  </label>
                  <label>
                    <span>Grouping</span>
                    <select
                      value={colmapVggtGrouping}
                      onChange={(event) =>
                        setColmapVggtGrouping(event.target.value as ColmapVggtGrouping)
                      }
                    >
                      <option value="sequential">Sequential (baseline)</option>
                      <option value="covisibility">Covisibility</option>
                    </select>
                  </label>
                  <label>
                    <span>Group overlap</span>
                    <input
                      type="number"
                      min={1}
                      max={7}
                      step={1}
                      value={colmapVggtOverlapSize}
                      onChange={(event) => setColmapVggtOverlapSize(Number(event.target.value))}
                    />
                  </label>
                  <label>
                    <span>Confidence</span>
                    <input
                      type="number"
                      min={0}
                      max={95}
                      step={5}
                      value={colmapVggtConfPercentile}
                      onChange={(event) => setColmapVggtConfPercentile(Number(event.target.value))}
                    />
                  </label>
                </div>

                <section className="ablation-controls" aria-labelledby="ablation-controls-title">
                  <div>
                    <strong id="ablation-controls-title">Dense-fusion ablations</strong>
                    <small>Baseline: global + any support + random. Each phase is independent.</small>
                  </div>
                  <label>
                    <span>Phase 1 · Confidence scope</span>
                    <select
                      value={confidenceThresholdScope}
                      onChange={(event) =>
                        setConfidenceThresholdScope(event.target.value as ConfidenceThresholdScope)
                      }
                    >
                      <option value="global">Global (baseline)</option>
                      <option value="per_frame">Per frame</option>
                    </select>
                  </label>
                  <label>
                    <span>Phase 2 · Consistency support</span>
                    <select
                      value={consistencySupportPolicy}
                      onChange={(event) =>
                        setConsistencySupportPolicy(event.target.value as ConsistencySupportPolicy)
                      }
                    >
                      <option value="any_support">Any support (baseline)</option>
                      <option value="adaptive_two">Adaptive two-view</option>
                    </select>
                  </label>
                  <label>
                    <span>Phase 3 · Point budget</span>
                    <select
                      value={pointBudgetPolicy}
                      onChange={(event) => setPointBudgetPolicy(event.target.value as PointBudgetPolicy)}
                    >
                      <option value="random">Random (baseline)</option>
                      <option value="spatial_balanced">Spatial balanced</option>
                    </select>
                  </label>
                </section>
              </>
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
          {currentStatus?.error && <div className="error-box">{currentStatus.error.message}</div>}

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
              <span>{viewerAsset ?? "No geometry loaded"}</span>
            </div>
            <div className="viewer-actions">
              {(hasPointCloud || hasMesh || hasSplat) && (
                <div className="variant-toggle" role="group" aria-label="Geometry preview">
                  {hasPointCloud && (
                    <button className={viewerMode === "point_cloud" ? "active" : ""} type="button" onClick={() => setViewerMode("point_cloud")}>
                      Point cloud
                    </button>
                  )}
                  {hasMesh && (
                    <button className={viewerMode === "mesh" ? "active" : ""} type="button" onClick={() => setViewerMode("mesh")}>
                      Mesh
                    </button>
                  )}
                  {hasSplat && (
                    <button
                      className={viewerMode === "gaussian_splat" ? "active" : ""}
                      type="button"
                      onClick={() => setViewerMode("gaussian_splat")}
                    >
                      Gaussian splat
                    </button>
                  )}
                </div>
              )}
              {hasPointCloud && viewerMode === "point_cloud" && (
                <div className="variant-toggle" role="group" aria-label="Point cloud view">
                  <button
                    className={pointCloudVariant === "raw" || !hasAlignedPointCloud ? "active" : ""}
                    type="button"
                    onClick={() => setPointCloudVariant("raw")}
                  >
                    Raw
                  </button>
                  <button
                    className={pointCloudVariant === "aligned" && hasAlignedPointCloud ? "active" : ""}
                    type="button"
                    onClick={() => setPointCloudVariant("aligned")}
                    disabled={!hasAlignedPointCloud}
                  >
                    Aligned
                  </button>
                </div>
              )}
              <button className="icon-button" type="button" onClick={refreshJob} disabled={!manifest}>
                <RefreshCw size={17} aria-hidden="true" />
                <span>Refresh</span>
              </button>
              {canCancel && (
                <button className="icon-button" type="button" onClick={() => changeLifecycle("cancel")} disabled={isChangingLifecycle}>
                  <Square size={16} aria-hidden="true" />
                  <span>Cancel</span>
                </button>
              )}
              {canRetry && (
                <button className="icon-button" type="button" onClick={() => changeLifecycle("retry")} disabled={isChangingLifecycle}>
                  <RotateCcw size={16} aria-hidden="true" />
                  <span>Retry</span>
                </button>
              )}
            </div>
          </div>
          <GeometryViewer
            pointCloudUrl={visiblePointCloudUrl}
            camerasUrl={viewerMode === "point_cloud" ? camerasUrl : null}
            alignmentDiagnosticsUrl={
              viewerMode === "point_cloud" ? alignmentDiagnosticsUrl : null
            }
            pointCloudVariant={pointCloudVariant}
            meshUrl={visibleMeshUrl}
            splatUrl={visibleSplatUrl}
            splatMetadataUrl={viewerMode === "gaussian_splat" ? splatMetadataUrl : null}
            splatCameraPathUrl={viewerMode === "gaussian_splat" ? splatCameraPathUrl : null}
          />
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
              <dt>Stage</dt>
              <dd>{currentStatus?.stage ?? "-"}</dd>
            </div>
            <div>
              <dt>Progress</dt>
              <dd>{currentStatus ? `${Math.round(currentStatus.progress * 100)}%` : "-"}</dd>
            </div>
            <div>
              <dt>Attempt</dt>
              <dd>{currentStatus?.active_attempt_id ?? "-"}</dd>
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
              <dt>Trainer</dt>
              <dd>{manifest?.gaussian_trainer?.label ?? "-"}</dd>
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
              <dd>{currentStatus?.metrics.batch_size ?? currentStatus?.metrics.vggt_batch_size ?? "-"}</dd>
            </div>
            <div>
              <dt>Grouping</dt>
              <dd>{formatPolicy(currentStatus?.metrics.vggt_grouping)}</dd>
            </div>
            <div>
              <dt>Overlap</dt>
              <dd>{currentStatus?.metrics.overlap_size ?? "-"}</dd>
            </div>
            <div>
              <dt>Alignment</dt>
              <dd>{currentStatus?.metrics.alignment_status ?? "-"}</dd>
            </div>
            <div>
              <dt>View check</dt>
              <dd>
                {currentStatus?.metrics.consistency_acceptance_rate === undefined
                  ? "-"
                  : `${(currentStatus.metrics.consistency_acceptance_rate * 100).toFixed(1)}%`}
              </dd>
            </div>
            <div>
              <dt>Rejected</dt>
              <dd>{currentStatus?.metrics.consistency_rejected ?? "-"}</dd>
            </div>
            <div>
              <dt>Residual P90</dt>
              <dd>
                {currentStatus?.metrics.consistency_residual_p90 === undefined
                  ? "-"
                  : `${(currentStatus.metrics.consistency_residual_p90 * 100).toFixed(2)}%`}
              </dd>
            </div>
            <div>
              <dt>Confidence scope</dt>
              <dd>{formatPolicy(currentStatus?.metrics.confidence_threshold_scope)}</dd>
            </div>
            <div>
              <dt>Support policy</dt>
              <dd>{formatPolicy(currentStatus?.metrics.consistency_support_policy)}</dd>
            </div>
            <div>
              <dt>Point budget</dt>
              <dd>{formatPolicy(currentStatus?.metrics.point_budget_policy)}</dd>
            </div>
            <div>
              <dt>Budget applied</dt>
              <dd>{formatBudgetApplied(currentStatus?.metrics.point_budget_applied)}</dd>
            </div>
            <div>
              <dt>Budget points</dt>
              <dd>
                {formatPointBudget(
                  currentStatus?.metrics.point_budget_input_points,
                  currentStatus?.metrics.point_budget_output_points
                )}
              </dd>
            </div>
            <div>
              <dt>Mesh</dt>
              <dd>{currentStatus?.metrics.mesh_status ?? "-"}</dd>
            </div>
            <div>
              <dt>Method</dt>
              <dd>{currentStatus?.metrics.mesh_method ?? "-"}</dd>
            </div>
            <div>
              <dt>Faces</dt>
              <dd>{currentStatus?.metrics.mesh_triangles ?? "-"}</dd>
            </div>
            <div>
              <dt>Components</dt>
              <dd>{currentStatus?.metrics.mesh_component_count ?? "-"}</dd>
            </div>
            <div>
              <dt>Trimmed</dt>
              <dd>{currentStatus?.metrics.mesh_long_edge_removed_triangles ?? "-"}</dd>
            </div>
          </dl>

          {canBuildMeshVariant && (
            <section className="result-section mesh-experiment">
              <div className="section-heading">
                <h3>Mesh variants</h3>
                <span>{meshVariants.length}</span>
              </div>

              <div className="control-stack mesh-settings">
                <label>
                  <span>Method</span>
                  <select
                    value={meshSettings.method}
                    onChange={(event) =>
                      setMeshSettings((current) => ({ ...current, method: event.target.value as MeshMethod }))
                    }
                  >
                    <option value="poisson">Poisson</option>
                    <option value="ball_pivoting">Ball pivoting</option>
                    <option value="alpha_shape">Alpha shape</option>
                  </select>
                </label>

                <div className="numeric-grid">
                  <NumberControl
                    label="Voxel"
                    min={0.005}
                    max={2}
                    step={0.005}
                    value={meshSettings.voxel_size}
                    onChange={(value) => updateMeshSetting("voxel_size", value)}
                  />
                  <NumberControl
                    label="Normal radius"
                    min={0.01}
                    max={5}
                    step={0.01}
                    value={meshSettings.normal_radius}
                    onChange={(value) => updateMeshSetting("normal_radius", value)}
                  />
                  <NumberControl
                    label="Noise std"
                    min={0.1}
                    max={10}
                    step={0.1}
                    value={meshSettings.statistical_std_ratio}
                    onChange={(value) => updateMeshSetting("statistical_std_ratio", value)}
                  />
                  <NumberControl
                    label="Edge factor"
                    min={0.5}
                    max={10}
                    step={0.1}
                    value={meshSettings.edge_trim_factor}
                    onChange={(value) => updateMeshSetting("edge_trim_factor", value)}
                  />
                  <NumberControl
                    label="Component ratio"
                    min={0}
                    max={0.5}
                    step={0.01}
                    value={meshSettings.component_min_ratio}
                    onChange={(value) => updateMeshSetting("component_min_ratio", value)}
                  />
                  <NumberControl
                    label="Max faces"
                    min={1_000}
                    max={1_000_000}
                    step={10_000}
                    value={meshSettings.max_triangles}
                    onChange={(value) => updateMeshSetting("max_triangles", value)}
                  />
                  {meshSettings.method === "poisson" && (
                    <>
                      <NumberControl
                        label="Poisson depth"
                        min={5}
                        max={12}
                        step={1}
                        value={meshSettings.poisson_depth}
                        onChange={(value) => updateMeshSetting("poisson_depth", value)}
                      />
                      <NumberControl
                        label="Density trim"
                        min={0}
                        max={0.9}
                        step={0.01}
                        value={meshSettings.density_trim_quantile}
                        onChange={(value) => updateMeshSetting("density_trim_quantile", value)}
                      />
                    </>
                  )}
                  {meshSettings.method === "alpha_shape" && (
                    <NumberControl
                      label="Alpha"
                      min={0}
                      max={10}
                      step={0.01}
                      value={meshSettings.alpha}
                      onChange={(value) => updateMeshSetting("alpha", value)}
                    />
                  )}
                </div>
              </div>

              <button className="primary-button" type="button" onClick={buildMeshVariant} disabled={isBuildingMesh}>
                <Play size={18} aria-hidden="true" />
                <span>{isBuildingMesh ? "Building mesh" : "Build mesh variant"}</span>
              </button>

              {meshVariants.length > 0 && (
                <div className="mesh-variant-list" role="group" aria-label="Mesh variants">
                  {meshVariants.map((variant) => (
                    <button
                      className={variant.id === selectedMeshVariant?.id ? "mesh-variant-row active" : "mesh-variant-row"}
                      key={variant.id}
                      type="button"
                      onClick={() => {
                        setSelectedMeshVariantId(variant.id);
                        setMeshSettings((current) => ({ ...current, ...variant.options, method: variant.method }));
                      }}
                    >
                      <strong>{variant.label}</strong>
                      <span>
                        {variant.metrics.mesh_triangles ?? "-"} faces · {variant.metrics.mesh_component_count ?? "-"} components
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </section>
          )}

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
              <AssetLink manifest={manifest} assetKey="point_cloud_aligned" label="Aligned point cloud" />
              <AssetLink manifest={manifest} assetKey="mesh" label="Mesh" />
              <AssetLink manifest={manifest} assetKey="mesh_diagnostics" label="Mesh diagnostics" />
              <AssetLink manifest={manifest} assetKey="scene_splat" label="Gaussian browser asset" />
              <AssetLink manifest={manifest} assetKey="gaussian_canonical" label="Canonical Gaussian PLY" />
              <AssetLink manifest={manifest} assetKey="gaussian_export_metadata" label="Gaussian export metadata" />
              <AssetLink manifest={manifest} assetKey="gaussian_evaluation" label="Gaussian validation" />
              <AssetLink manifest={manifest} assetKey="gaussian_test_evaluation" label="Gaussian held-out test" />
              <AssetLink manifest={manifest} assetKey="gaussian_test_decision" label="Gaussian test decision" />
              <AssetLink manifest={manifest} assetKey="gaussian_camera_path" label="Gaussian camera path" />
              <AssetLink manifest={manifest} assetKey="gaussian_bundle" label="Gaussian result bundle" />
              <AssetLink manifest={manifest} assetKey="alignment_diagnostics" label="Alignment" />
              <AssetLink manifest={manifest} assetKey="fusion_diagnostics" label="Fusion diagnostics" />
              <AssetLink manifest={manifest} assetKey="visibility_graph" label="Visibility graph" />
              <AssetLink manifest={manifest} assetKey="consistency_diagnostics" label="Consistency diagnostics" />
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

function NumberControl({
  label,
  min,
  max,
  step,
  value,
  onChange
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label>
      <span>{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
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

function validateColmapVggtOptions(
  batchSize: number,
  overlapSize: number,
  maxPoints: number,
  confPercentile: number
) {
  if (!Number.isInteger(batchSize) || batchSize < 2) {
    return "COLMAP+VGGT depth batch must be an integer of at least 2.";
  }
  if (!Number.isInteger(overlapSize) || overlapSize <= 0) {
    return "COLMAP+VGGT group overlap must be a positive integer.";
  }
  if (overlapSize >= batchSize) {
    return "COLMAP+VGGT group overlap must be smaller than depth batch.";
  }
  if (!Number.isInteger(maxPoints) || maxPoints <= 0) {
    return "COLMAP+VGGT max points must be a positive integer.";
  }
  if (!Number.isFinite(confPercentile) || confPercentile < 0 || confPercentile >= 100) {
    return "COLMAP+VGGT confidence must be between 0 and 99.";
  }
  return null;
}

function validateMeshSettings(settings: MeshSettings) {
  const values = Object.entries(settings).filter(([key]) => key !== "method");
  if (values.some(([, value]) => typeof value !== "number" || !Number.isFinite(value))) {
    return "Mesh settings must be finite numbers.";
  }
  if (!Number.isInteger(settings.poisson_depth) || !Number.isInteger(settings.max_triangles)) {
    return "Poisson depth and max faces must be integers.";
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

function formatPolicy(value: string | undefined) {
  return value?.replaceAll("_", " ") ?? "-";
}

function formatBudgetApplied(value: boolean | undefined) {
  if (value === undefined) {
    return "-";
  }
  return value ? "yes" : "no";
}

function formatPointBudget(inputPoints: number | undefined, outputPoints: number | undefined) {
  if (inputPoints === undefined || outputPoints === undefined) {
    return "-";
  }
  return `${inputPoints.toLocaleString()} → ${outputPoints.toLocaleString()}`;
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
