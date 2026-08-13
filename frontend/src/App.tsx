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
import companyLogo from "./assets/yuetron-logo.png";
import { isBackendAvailable, isOutputSupported } from "./backendOptions";
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
    collision_mesh?: string;
    navigation?: string;
    navigation_diagnostics?: string;
    scene_graph?: string;
    log?: string;
  };
  gaussian_trainer?: {
    id: GaussianTrainer;
    label: string;
    revision: string;
    license: string;
  };
  gaussian_config?: {
    schema_version: number;
    requested_profile: string;
    effective_config_hash: string;
  };
  mesh_variants?: MeshVariant[];
  navigation_status?: "pending" | "not_generated" | "queued" | "generating" | "available" | "unavailable";
  navigation_reason?: string | null;
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

type JobSummary = {
  job_id: string;
  status: string;
  geometry_backend: GeometryBackend;
  output_type: OutputType;
  updated_at: string;
};

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
  { id: "image", label: "单张图片", icon: Image, fileHint: "上传 1 张图片" },
  { id: "multi_image", label: "多张图片", icon: Images, fileHint: "上传至少 2 张图片" },
  { id: "video", label: "视频", icon: Video, fileHint: "上传 1 个视频" },
  { id: "panorama", label: "全景图", icon: FileArchive, fileHint: "上传 1 张 360° 全景图" }
];

const backendOptions: Array<{
  id: GeometryBackend;
  label: string;
}> = [
  { id: "mock", label: "模拟后端（Mock）" },
  { id: "vggt", label: "VGGT（视觉几何基础模型）" },
  { id: "colmap", label: "COLMAP（摄影测量重建）" },
  { id: "colmap_vggt", label: "COLMAP + VGGT（融合重建）" },
  { id: "dust3r", label: "DUSt3R（稠密三维重建）" },
  { id: "mast3r", label: "MASt3R（匹配与三维重建）" },
  { id: "project_3dgs", label: "Project 3DGS（项目高斯重建）" }
];

const outputOptions: Array<{
  id: OutputType;
  label: string;
}> = [
  { id: "point_cloud", label: "点云（Point Cloud）" },
  { id: "mesh", label: "网格（Mesh）" },
  { id: "gaussian_splat", label: "高斯泼溅（Gaussian Splat）" }
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
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [selectedJobId, setSelectedJobId] = useState("");
  const [pointCloudVariant, setPointCloudVariant] = useState<"raw" | "aligned">("aligned");
  const [viewerMode, setViewerMode] = useState<ViewerMode>("point_cloud");
  const [meshSettings, setMeshSettings] = useState<MeshSettings>(defaultMeshSettings);
  const [selectedMeshVariantId, setSelectedMeshVariantId] = useState<string | null>(null);
  const [isBuildingMesh, setIsBuildingMesh] = useState(false);
  const [isBuildingNavigation, setIsBuildingNavigation] = useState(false);
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
  const selectedBackendAvailable = selectedBackendStatus?.available ?? true;
  const selectedOutputSupported =
    selectedBackendStatus?.supported_outputs.includes(outputType) ?? true;
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
  const collisionMeshUrl = useMemo(() => {
    if (!manifest?.assets.collision_mesh) {
      return null;
    }
    return `/api/jobs/${manifest.job_id}/assets/${manifest.assets.collision_mesh}`;
  }, [manifest]);
  const navigationUrl = useMemo(() => {
    if (!manifest?.assets.navigation) {
      return null;
    }
    return `/api/jobs/${manifest.job_id}/assets/${manifest.assets.navigation}`;
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

    requestJson<{ jobs: JobSummary[] }>("/api/jobs")
      .then((payload) => {
        if (!cancelled) {
          setJobs(payload.jobs);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setJobs([]);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (
      !manifest ||
      (!["queued", "running", "exporting"].includes(manifest.status) &&
        !["queued", "generating"].includes(manifest.navigation_status ?? ""))
    ) {
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
          setError(caught instanceof Error ? caught.message : "刷新任务失败");
        }
      }
    }, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [manifest?.job_id, manifest?.status, manifest?.navigation_status]);

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
      setError(selectedBackendStatus?.reason ?? "所选几何重建后端不可用。");
      return;
    }
    if (!selectedOutputSupported) {
      setError("所选重建后端不支持当前输出类型。");
      return;
    }
    if (geometryBackend === "project_3dgs" && selectedGaussianTrainerStatus?.available === false) {
      setError(selectedGaussianTrainerStatus.reason ?? "所选高斯训练器不可用。");
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
      setJobs((current) => [
        {
          job_id: created.job_id,
          status: created.status,
          geometry_backend: created.geometry_backend,
          output_type: created.output_type,
          updated_at: created.updated_at ?? created.created_at
        },
        ...current.filter((job) => job.job_id !== created.job_id)
      ]);
      setSelectedJobId(created.job_id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "创建任务失败");
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
      setError(caught instanceof Error ? caught.message : "刷新任务失败");
    }
  }

  async function loadJobById(jobId: string) {
    if (!jobId) {
      return;
    }

    setSelectedJobId(jobId);
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
      setError(caught instanceof Error ? caught.message : "加载任务失败");
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

  async function buildNavigationAssets() {
    if (!manifest) {
      return;
    }
    setIsBuildingNavigation(true);
    setError(null);
    try {
      const nextManifest = await requestJson<Manifest>(
        `/api/jobs/${manifest.job_id}/navigation-assets`,
        { method: "POST" }
      );
      applyManifest(nextManifest);
      setJobStatus(nextManifest);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "生成导航资产失败");
    } finally {
      setIsBuildingNavigation(false);
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
      setError(caught instanceof Error ? caught.message : "生成网格方案失败");
    } finally {
      setIsBuildingMesh(false);
    }
  }

  const currentStatus = jobStatus ?? manifest;
  const canCancel = Boolean(currentStatus && ["queued", "running", "exporting"].includes(currentStatus.status));
  const canRetry = Boolean(currentStatus && ["failed", "cancelled"].includes(currentStatus.status));
  const hasAlignedPointCloud = Boolean(manifest?.assets.point_cloud_aligned);
  const canBuildMeshVariant = Boolean(manifest?.assets.point_cloud || manifest?.assets.point_cloud_aligned);
  const canBuildNavigation = Boolean(
    manifest?.status === "done" &&
      manifest.geometry_backend === "project_3dgs" &&
      manifest.output_type === "gaussian_splat" &&
      manifest.navigation_status !== "available" &&
      !["queued", "generating"].includes(manifest.navigation_status ?? "")
  );

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <img className="company-logo" src={companyLogo} alt="越创智数 YUETRON DIGTECH" />
          <div className="brand-divider" aria-hidden="true" />
          <div>
            <p className="eyebrow">Image3D-SceneGraph · 图像三维场景图</p>
            <h1>智能三维场景重建平台</h1>
          </div>
        </div>
        <div className="status-chip">{formatStatus(currentStatus?.status)}</div>
      </header>

      <section className="workspace">
        <aside className="panel upload-panel" aria-label="任务输入">
          <div className="panel-heading">
            <h2>数据输入</h2>
            <span>{selectedMode.fileHint}</span>
          </div>

          <div className="mode-grid" role="group" aria-label="重建模式">
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
              <span>几何重建后端</span>
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
              <span>输出类型</span>
              <select
                value={outputType}
                onChange={(event) => setOutputType(event.target.value as OutputType)}
              >
                {outputOptions.map((option) => (
                  <option disabled={!isOutputSupported(option.id, selectedBackendStatus)} key={option.id} value={option.id}>
                    {isOutputSupported(option.id, selectedBackendStatus) ? option.label : `${option.label}（不可用）`}
                  </option>
                ))}
              </select>
            </label>

            {geometryBackend === "project_3dgs" && (
              <label>
                <span>训练器（Trainer）</span>
                <select
                  value={gaussianTrainer}
                  onChange={(event) => setGaussianTrainer(event.target.value as GaussianTrainer)}
                >
                  {(gaussianTrainerStatuses.length > 0
                    ? gaussianTrainerStatuses
                    : [
                        {
                          id: "graphdeco" as const,
                          label: "Graphdeco 官方训练器",
                          available: true,
                          reason: null,
                          setup_command: null,
                          revision: "unknown",
                          license: "仅限 Graphdeco 研究与评估"
                        },
                        {
                          id: "project" as const,
                          label: "Project v7（gsplat 高斯栅格化）",
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
                {gaussianTrainer === "project" && (
                  <small>
                    固定 v7 配置：30,000 次迭代上限 · 最长边 1280px · 3NN（三近邻）初始化 ·
                    关闭屏幕半径剪枝 · 由验证集选择模型 · 归一化任意单位。
                  </small>
                )}
              </label>
            )}

            {geometryBackend === "vggt" && (
              <div className="numeric-grid">
                <label>
                  <span>最大图片数</span>
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
                  <span>批次大小（Batch Size）</span>
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
                  <span>重叠数量（Overlap）</span>
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
                    <span>最大点数</span>
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
                    <span>深度批次</span>
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
                    <span>分组策略</span>
                    <select
                      value={colmapVggtGrouping}
                      onChange={(event) =>
                        setColmapVggtGrouping(event.target.value as ColmapVggtGrouping)
                      }
                    >
                      <option value="sequential">顺序分组（基线）</option>
                      <option value="covisibility">共视分组（Covisibility）</option>
                    </select>
                  </label>
                  <label>
                    <span>分组重叠数</span>
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
                    <span>置信度百分位</span>
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
                    <strong id="ablation-controls-title">稠密融合消融实验</strong>
                    <small>基线：全局阈值 + 任意视图支持 + 随机预算；各阶段相互独立。</small>
                  </div>
                  <label>
                    <span>阶段 1 · 置信度范围</span>
                    <select
                      value={confidenceThresholdScope}
                      onChange={(event) =>
                        setConfidenceThresholdScope(event.target.value as ConfidenceThresholdScope)
                      }
                    >
                      <option value="global">全局阈值（基线）</option>
                      <option value="per_frame">逐帧阈值</option>
                    </select>
                  </label>
                  <label>
                    <span>阶段 2 · 一致性支持</span>
                    <select
                      value={consistencySupportPolicy}
                      onChange={(event) =>
                        setConsistencySupportPolicy(event.target.value as ConsistencySupportPolicy)
                      }
                    >
                      <option value="any_support">任意视图支持（基线）</option>
                      <option value="adaptive_two">自适应双视图</option>
                    </select>
                  </label>
                  <label>
                    <span>阶段 3 · 点数预算</span>
                    <select
                      value={pointBudgetPolicy}
                      onChange={(event) => setPointBudgetPolicy(event.target.value as PointBudgetPolicy)}
                    >
                      <option value="random">随机采样（基线）</option>
                      <option value="spatial_balanced">空间均衡采样</option>
                    </select>
                  </label>
                </section>
              </>
            )}
          </div>

          <div className="file-picker-grid">
            <label className="file-drop">
              <UploadCloud size={22} aria-hidden="true" />
              <span>{files.length > 0 ? `已选择 ${files.length} 个文件` : "选择文件"}</span>
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
                <span>选择文件夹</span>
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
              <p>尚未选择文件</p>
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
            <span>{isSubmitting ? "正在创建任务" : "创建重建任务"}</span>
          </button>
        </aside>

        <section className="viewer-column">
          <div className="viewer-header">
            <div>
              <h2>三维查看器（3D Viewer）</h2>
              <span>{viewerAsset ?? "尚未加载几何资产"}</span>
            </div>
            <div className="viewer-actions">
              {(hasPointCloud || hasMesh || hasSplat) && (
                <div className="variant-toggle" role="group" aria-label="几何预览类型">
                  {hasPointCloud && (
                    <button className={viewerMode === "point_cloud" ? "active" : ""} type="button" onClick={() => setViewerMode("point_cloud")}>
                      点云（Point Cloud）
                    </button>
                  )}
                  {hasMesh && (
                    <button className={viewerMode === "mesh" ? "active" : ""} type="button" onClick={() => setViewerMode("mesh")}>
                      网格（Mesh）
                    </button>
                  )}
                  {hasSplat && (
                    <button
                      className={viewerMode === "gaussian_splat" ? "active" : ""}
                      type="button"
                      onClick={() => setViewerMode("gaussian_splat")}
                    >
                      高斯泼溅（Gaussian Splat）
                    </button>
                  )}
                </div>
              )}
              {hasPointCloud && viewerMode === "point_cloud" && (
                <div className="variant-toggle" role="group" aria-label="点云视图">
                  <button
                    className={pointCloudVariant === "raw" || !hasAlignedPointCloud ? "active" : ""}
                    type="button"
                    onClick={() => setPointCloudVariant("raw")}
                  >
                    原始点云
                  </button>
                  <button
                    className={pointCloudVariant === "aligned" && hasAlignedPointCloud ? "active" : ""}
                    type="button"
                    onClick={() => setPointCloudVariant("aligned")}
                    disabled={!hasAlignedPointCloud}
                  >
                    对齐点云
                  </button>
                </div>
              )}
              <button className="icon-button" type="button" onClick={refreshJob} disabled={!manifest}>
                <RefreshCw size={17} aria-hidden="true" />
                <span>刷新</span>
              </button>
              {canCancel && (
                <button className="icon-button" type="button" onClick={() => changeLifecycle("cancel")} disabled={isChangingLifecycle}>
                  <Square size={16} aria-hidden="true" />
                  <span>取消</span>
                </button>
              )}
              {canRetry && (
                <button className="icon-button" type="button" onClick={() => changeLifecycle("retry")} disabled={isChangingLifecycle}>
                  <RotateCcw size={16} aria-hidden="true" />
                  <span>重试</span>
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
            collisionMeshUrl={viewerMode === "gaussian_splat" ? collisionMeshUrl : null}
            navigationUrl={viewerMode === "gaussian_splat" ? navigationUrl : null}
            navigationStatus={manifest?.navigation_status ?? null}
            navigationReason={manifest?.navigation_reason ?? null}
          />
        </section>

        <aside className="panel result-panel" aria-label="任务结果">
          <div className="panel-heading">
            <h2>任务信息</h2>
            <span>{manifest?.job_id ?? "暂无任务"}</span>
          </div>

          <div className="load-job-row">
            <select
              aria-label="选择历史任务"
              value={selectedJobId}
              disabled={isLoadingJob || jobs.length === 0}
              onChange={(event) => void loadJobById(event.target.value)}
            >
              <option value="">
                {jobs.length === 0 ? "未发现有效任务" : "选择历史任务"}
              </option>
              {jobs.map((job) => (
                <option key={job.job_id} value={job.job_id}>
                  {formatJobOption(job)}
                </option>
              ))}
            </select>
            <span className="job-load-status">{isLoadingJob ? "加载中…" : `${jobs.length} 个任务`}</span>
          </div>

          <dl className="metrics-grid">
            <div>
              <dt>状态</dt>
              <dd>{formatStatus(currentStatus?.status)}</dd>
            </div>
            <div>
              <dt>处理阶段</dt>
              <dd>{formatStage(currentStatus?.stage)}</dd>
            </div>
            <div>
              <dt>进度</dt>
              <dd>{currentStatus ? `${Math.round(currentStatus.progress * 100)}%` : "-"}</dd>
            </div>
            <div>
              <dt>运行批次</dt>
              <dd>{currentStatus?.active_attempt_id ?? "-"}</dd>
            </div>
            <div>
              <dt>输入模式</dt>
              <dd>{formatMode(currentStatus?.mode)}</dd>
            </div>
            <div>
              <dt>重建后端</dt>
              <dd>{formatBackend(currentStatus?.geometry_backend)}</dd>
            </div>
            <div>
              <dt>输出类型</dt>
              <dd>{formatOutput(currentStatus?.output_type)}</dd>
            </div>
            <div>
              <dt>训练器</dt>
              <dd>{formatTrainer(manifest?.gaussian_trainer)}</dd>
            </div>
            <div>
              <dt>高斯配置</dt>
              <dd>
                {manifest?.gaussian_config
                  ? `${manifest.gaussian_config.requested_profile} · v${manifest.gaussian_config.schema_version}`
                  : "-"}
              </dd>
            </div>
            <div>
              <dt>导航资产</dt>
              <dd>{formatStatus(manifest?.navigation_status)}</dd>
            </div>
            <div>
              <dt>输入数量</dt>
              <dd>{currentStatus?.metrics.num_inputs ?? "-"}</dd>
            </div>
            <div>
              <dt>点数量</dt>
              <dd>{currentStatus?.metrics.num_points ?? "-"}</dd>
            </div>
            <div>
              <dt>分组数量</dt>
              <dd>{currentStatus?.metrics.num_groups ?? "-"}</dd>
            </div>
            <div>
              <dt>批次大小</dt>
              <dd>{currentStatus?.metrics.batch_size ?? currentStatus?.metrics.vggt_batch_size ?? "-"}</dd>
            </div>
            <div>
              <dt>分组策略</dt>
              <dd>{formatPolicy(currentStatus?.metrics.vggt_grouping)}</dd>
            </div>
            <div>
              <dt>重叠数量</dt>
              <dd>{currentStatus?.metrics.overlap_size ?? "-"}</dd>
            </div>
            <div>
              <dt>空间对齐</dt>
              <dd>{formatPolicy(currentStatus?.metrics.alignment_status)}</dd>
            </div>
            <div>
              <dt>多视图通过率</dt>
              <dd>
                {currentStatus?.metrics.consistency_acceptance_rate === undefined
                  ? "-"
                  : `${(currentStatus.metrics.consistency_acceptance_rate * 100).toFixed(1)}%`}
              </dd>
            </div>
            <div>
              <dt>剔除数量</dt>
              <dd>{currentStatus?.metrics.consistency_rejected ?? "-"}</dd>
            </div>
            <div>
              <dt>残差 P90（90%分位）</dt>
              <dd>
                {currentStatus?.metrics.consistency_residual_p90 === undefined
                  ? "-"
                  : `${(currentStatus.metrics.consistency_residual_p90 * 100).toFixed(2)}%`}
              </dd>
            </div>
            <div>
              <dt>置信度范围</dt>
              <dd>{formatPolicy(currentStatus?.metrics.confidence_threshold_scope)}</dd>
            </div>
            <div>
              <dt>支持策略</dt>
              <dd>{formatPolicy(currentStatus?.metrics.consistency_support_policy)}</dd>
            </div>
            <div>
              <dt>点数预算</dt>
              <dd>{formatPolicy(currentStatus?.metrics.point_budget_policy)}</dd>
            </div>
            <div>
              <dt>是否应用预算</dt>
              <dd>{formatBudgetApplied(currentStatus?.metrics.point_budget_applied)}</dd>
            </div>
            <div>
              <dt>预算前后点数</dt>
              <dd>
                {formatPointBudget(
                  currentStatus?.metrics.point_budget_input_points,
                  currentStatus?.metrics.point_budget_output_points
                )}
              </dd>
            </div>
            <div>
              <dt>网格状态</dt>
              <dd>{formatPolicy(currentStatus?.metrics.mesh_status)}</dd>
            </div>
            <div>
              <dt>网格方法</dt>
              <dd>{formatPolicy(currentStatus?.metrics.mesh_method)}</dd>
            </div>
            <div>
              <dt>三角面数量</dt>
              <dd>{currentStatus?.metrics.mesh_triangles ?? "-"}</dd>
            </div>
            <div>
              <dt>连通组件</dt>
              <dd>{currentStatus?.metrics.mesh_component_count ?? "-"}</dd>
            </div>
            <div>
              <dt>裁剪面数量</dt>
              <dd>{currentStatus?.metrics.mesh_long_edge_removed_triangles ?? "-"}</dd>
            </div>
          </dl>

          {manifest?.status === "done" &&
            manifest.geometry_backend === "project_3dgs" &&
            manifest.output_type === "gaussian_splat" && (
              <section className="result-section">
                <div className="section-heading">
                  <h3>第一人称导航</h3>
                  <span>{formatStatus(manifest.navigation_status)}</span>
                </div>
                {manifest.navigation_status === "available" ? (
                  <p className="empty-text">打开高斯泼溅视图，并选择“进入漫游”。</p>
                ) : (
                  <>
                    <p className="empty-text">
                      {manifest.navigation_reason
                        ? `不可用：${formatPolicy(manifest.navigation_reason)}`
                        : "无需重新训练，使用训练集（Train）生成碰撞体与边界资产。"}
                    </p>
                    <button
                      className="primary-button"
                      type="button"
                      onClick={buildNavigationAssets}
                      disabled={!canBuildNavigation || isBuildingNavigation}
                    >
                      <Play size={18} aria-hidden="true" />
                      <span>
                        {["queued", "generating"].includes(manifest.navigation_status ?? "")
                          ? "正在生成导航"
                          : isBuildingNavigation
                            ? "正在加入导航队列"
                            : "生成导航资产"}
                      </span>
                    </button>
                  </>
                )}
              </section>
            )}

          {canBuildMeshVariant && (
            <section className="result-section mesh-experiment">
              <div className="section-heading">
                <h3>网格方案（Mesh Variants）</h3>
                <span>{meshVariants.length}</span>
              </div>

              <div className="control-stack mesh-settings">
                <label>
                  <span>生成方法</span>
                  <select
                    value={meshSettings.method}
                    onChange={(event) =>
                      setMeshSettings((current) => ({ ...current, method: event.target.value as MeshMethod }))
                    }
                  >
                    <option value="poisson">泊松重建（Poisson）</option>
                    <option value="ball_pivoting">滚动球法（Ball Pivoting）</option>
                    <option value="alpha_shape">Alpha Shape（α 形状）</option>
                  </select>
                </label>

                <div className="numeric-grid">
                  <NumberControl
                    label="体素大小（Voxel）"
                    min={0.005}
                    max={2}
                    step={0.005}
                    value={meshSettings.voxel_size}
                    onChange={(value) => updateMeshSetting("voxel_size", value)}
                  />
                  <NumberControl
                    label="法线半径"
                    min={0.01}
                    max={5}
                    step={0.01}
                    value={meshSettings.normal_radius}
                    onChange={(value) => updateMeshSetting("normal_radius", value)}
                  />
                  <NumberControl
                    label="噪声标准差"
                    min={0.1}
                    max={10}
                    step={0.1}
                    value={meshSettings.statistical_std_ratio}
                    onChange={(value) => updateMeshSetting("statistical_std_ratio", value)}
                  />
                  <NumberControl
                    label="边缘裁剪系数"
                    min={0.5}
                    max={10}
                    step={0.1}
                    value={meshSettings.edge_trim_factor}
                    onChange={(value) => updateMeshSetting("edge_trim_factor", value)}
                  />
                  <NumberControl
                    label="组件最小占比"
                    min={0}
                    max={0.5}
                    step={0.01}
                    value={meshSettings.component_min_ratio}
                    onChange={(value) => updateMeshSetting("component_min_ratio", value)}
                  />
                  <NumberControl
                    label="最大三角面数"
                    min={1_000}
                    max={1_000_000}
                    step={10_000}
                    value={meshSettings.max_triangles}
                    onChange={(value) => updateMeshSetting("max_triangles", value)}
                  />
                  {meshSettings.method === "poisson" && (
                    <>
                      <NumberControl
                        label="泊松深度"
                        min={5}
                        max={12}
                        step={1}
                        value={meshSettings.poisson_depth}
                        onChange={(value) => updateMeshSetting("poisson_depth", value)}
                      />
                      <NumberControl
                        label="密度裁剪分位"
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
                      label="Alpha（α）"
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
                <span>{isBuildingMesh ? "正在生成网格" : "生成网格方案"}</span>
              </button>

              {meshVariants.length > 0 && (
                <div className="mesh-variant-list" role="group" aria-label="网格方案">
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
                        {variant.metrics.mesh_triangles ?? "-"} 个三角面 · {variant.metrics.mesh_component_count ?? "-"} 个组件
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </section>
          )}

          <section className="result-section">
            <h3>语义对象</h3>
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
              <p className="empty-text">暂无语义对象</p>
            )}
          </section>

          <section className="result-section">
            <h3>结果资产</h3>
            <div className="asset-links">
              <AssetLink manifest={manifest} assetKey="point_cloud" label="点云（Point Cloud）" />
              <AssetLink manifest={manifest} assetKey="point_cloud_aligned" label="对齐点云" />
              <AssetLink manifest={manifest} assetKey="mesh" label="三维网格（Mesh）" />
              <AssetLink manifest={manifest} assetKey="mesh_diagnostics" label="网格诊断" />
              <AssetLink manifest={manifest} assetKey="scene_splat" label="高斯浏览资产" />
              <AssetLink manifest={manifest} assetKey="gaussian_canonical" label="标准高斯 PLY" />
              <AssetLink manifest={manifest} assetKey="gaussian_export_metadata" label="高斯导出元数据" />
              <AssetLink manifest={manifest} assetKey="gaussian_evaluation" label="高斯验证集评估" />
              <AssetLink manifest={manifest} assetKey="gaussian_test_evaluation" label="高斯留出测试集评估" />
              <AssetLink manifest={manifest} assetKey="gaussian_test_decision" label="高斯测试判定" />
              <AssetLink manifest={manifest} assetKey="gaussian_camera_path" label="高斯相机路径" />
              <AssetLink manifest={manifest} assetKey="gaussian_bundle" label="高斯结果包" />
              <AssetLink manifest={manifest} assetKey="collision_mesh" label="碰撞网格" />
              <AssetLink manifest={manifest} assetKey="navigation" label="导航数据约定" />
              <AssetLink manifest={manifest} assetKey="navigation_diagnostics" label="导航诊断" />
              <AssetLink manifest={manifest} assetKey="alignment_diagnostics" label="空间对齐诊断" />
              <AssetLink manifest={manifest} assetKey="fusion_diagnostics" label="融合诊断" />
              <AssetLink manifest={manifest} assetKey="visibility_graph" label="可见性图" />
              <AssetLink manifest={manifest} assetKey="consistency_diagnostics" label="一致性诊断" />
              <AssetLink manifest={manifest} assetKey="scene_graph" label="场景图（Scene Graph）" />
              <AssetLink manifest={manifest} assetKey="log" label="运行日志" />
              {manifest && (
                <a href={`/api/jobs/${manifest.job_id}/download`}>
                  <Download size={16} aria-hidden="true" />
                  <span>下载完整结果包</span>
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
    return "请至少选择一个文件。";
  }
  if ((mode === "image" || mode === "video" || mode === "panorama") && files.length !== 1) {
    return "当前模式只能上传一个文件。";
  }
  if (mode === "multi_image" && files.length < 2) {
    return "多图模式至少需要两个文件。";
  }
  return null;
}

function validateVggtOptions(maxImages: number, batchSize: number, overlapSize: number, fileCount: number) {
  if (!Number.isInteger(maxImages) || maxImages <= 0) {
    return "VGGT 最大图片数必须是正整数。";
  }
  if (!Number.isInteger(batchSize) || batchSize <= 0) {
    return "VGGT 批次大小必须是正整数。";
  }
  if (!Number.isInteger(overlapSize) || overlapSize <= 0) {
    return "VGGT 重叠数量必须是正整数。";
  }
  if (fileCount > 1 && batchSize < 2) {
    return "多图任务的 VGGT 批次大小至少为 2。";
  }
  if (overlapSize >= batchSize) {
    return "VGGT 重叠数量必须小于批次大小。";
  }
  if (batchSize > maxImages) {
    return "VGGT 批次大小不能超过最大图片数。";
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
    return "COLMAP+VGGT 深度批次必须是至少为 2 的整数。";
  }
  if (!Number.isInteger(overlapSize) || overlapSize <= 0) {
    return "COLMAP+VGGT 分组重叠数必须是正整数。";
  }
  if (overlapSize >= batchSize) {
    return "COLMAP+VGGT 分组重叠数必须小于深度批次。";
  }
  if (!Number.isInteger(maxPoints) || maxPoints <= 0) {
    return "COLMAP+VGGT 最大点数必须是正整数。";
  }
  if (!Number.isFinite(confPercentile) || confPercentile < 0 || confPercentile >= 100) {
    return "COLMAP+VGGT 置信度必须在 0 到 99 之间。";
  }
  return null;
}

function validateMeshSettings(settings: MeshSettings) {
  const values = Object.entries(settings).filter(([key]) => key !== "method");
  if (values.some(([, value]) => typeof value !== "number" || !Number.isFinite(value))) {
    return "网格参数必须是有效数值。";
  }
  if (!Number.isInteger(settings.poisson_depth) || !Number.isInteger(settings.max_triangles)) {
    return "泊松深度与最大三角面数必须是整数。";
  }
  return null;
}

function formatBackendOption(
  option: { id: GeometryBackend; label: string },
  statuses: Record<GeometryBackend, BackendStatus> | null
) {
  const status = statuses?.[option.id];
  if (!status) {
    return option.id === "mock" ? option.label : `${option.label}（检查中）`;
  }
  return status.available ? option.label : `${option.label}（需要安装）`;
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

function formatJobOption(job: JobSummary) {
  return `${job.job_id} · ${formatStatus(job.status)} · ${formatBackend(job.geometry_backend)} / ${formatOutput(job.output_type)}`;
}

function formatStatus(value: string | null | undefined) {
  const labels: Record<string, string> = {
    idle: "空闲",
    pending: "等待中",
    queued: "已排队",
    running: "处理中",
    exporting: "正在导出",
    done: "已完成",
    failed: "失败",
    cancelled: "已取消",
    not_generated: "未生成",
    generating: "正在生成",
    available: "可用",
    unavailable: "不可用"
  };
  return value ? (labels[value] ?? formatPolicy(value)) : "-";
}

function formatStage(value: string | undefined) {
  const labels: Record<string, string> = {
    queued: "等待执行",
    validating: "校验输入",
    reconstructing: "几何重建",
    training: "高斯训练",
    evaluating: "质量评估",
    exporting: "导出结果",
    postprocessing: "后处理",
    done: "处理完成",
    failed: "处理失败"
  };
  return value ? (labels[value] ?? formatPolicy(value)) : "-";
}

function formatMode(value: Mode | undefined) {
  return modeOptions.find((option) => option.id === value)?.label ?? "-";
}

function formatBackend(value: GeometryBackend | undefined) {
  return backendOptions.find((option) => option.id === value)?.label ?? "-";
}

function formatOutput(value: OutputType | undefined) {
  return outputOptions.find((option) => option.id === value)?.label ?? "-";
}

function formatTrainer(trainer: Manifest["gaussian_trainer"] | undefined) {
  if (!trainer) {
    return "-";
  }
  return trainer.id === "project"
    ? "Project v7（gsplat 高斯栅格化）"
    : "Graphdeco 官方训练器（研究与评估）";
}

function formatPolicy(value: string | undefined) {
  const labels: Record<string, string> = {
    sequential: "顺序分组（基线）",
    covisibility: "共视分组（Covisibility）",
    global: "全局阈值",
    per_frame: "逐帧阈值",
    any_support: "任意视图支持",
    adaptive_two: "自适应双视图",
    random: "随机采样",
    spatial_balanced: "空间均衡采样",
    complete: "完成",
    skipped: "已跳过",
    not_run: "未运行",
    failed: "失败",
    unavailable: "不可用",
    poisson: "泊松重建（Poisson）",
    ball_pivoting: "滚动球法（Ball Pivoting）",
    alpha_shape: "Alpha Shape（α 形状）"
  };
  return value ? (labels[value] ?? value.replaceAll("_", " ")) : "-";
}

function formatBudgetApplied(value: boolean | undefined) {
  if (value === undefined) {
    return "-";
  }
  return value ? "是" : "否";
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
