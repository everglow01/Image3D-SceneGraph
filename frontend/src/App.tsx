import { ChangeEvent, useEffect, useMemo, useState } from "react";
import {
  Download,
  FileArchive,
  Image,
  Images,
  Maximize2,
  Minimize2,
  Play,
  RefreshCw,
  RotateCcw,
  Square,
  UploadCloud,
  Video
} from "lucide-react";
import companyLogo from "./assets/yuetron-logo.png";
import {
  isBackendAvailable,
  isOutputSupported,
  type ExperimentalOptionStatus
} from "./backendOptions";
import { GeometryViewer } from "./GeometryViewer";
import { ReconstructionEvidenceRail } from "./ReconstructionEvidenceRail";
import {
  buildEvidenceStages,
  type EvidenceStageId
} from "./reconstructionEvidence";
import type { SfmInspectionTab } from "./sfmDiagnostics";
import {
  defaultSfmCameraCalibration,
  formatSfmCameraCalibration,
  formatSfmFeatureProfile,
  formatSfmGeometricVerification,
  formatSfmLocalMatcher,
  formatSfmPairing,
  isSfmCameraCalibrationAvailable,
  isSfmGeometricVerificationAvailable,
  isSfmPairingAvailable,
  isSfmPairingModeSupported,
  sfmCameraCalibrationOptions,
  sfmFeatureOptions,
  sfmGeometricVerificationOptions,
  sfmLocalMatcherOptions,
  sfmPairingOptions
} from "./sfmOptions";
import type {
  SfmCameraCalibration,
  SfmCameraCalibrationStatus,
  SfmFeatureProfile,
  SfmFeatureStatus,
  SfmGeometricVerification,
  SfmLocalMatcher,
  SfmPairing
} from "./sfmOptions";
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
type VideoRotation = "auto" | "clockwise_90" | "counterclockwise_90" | "180";
type GaussianGeometrySource = "colmap" | "vggt_ba";
type GaussianPostprocess = "none" | "vggt_visibility_v1";
type GaussianSorFilter = "on" | "off";
type GaussianVariant = "original" | "vggt_filtered";

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
    sfm_sparse_point_cloud?: string;
    sfm_diagnostics?: string;
    sfm_camera_calibration_diagnostics?: string;
    sfm_pose_health?: string;
    sfm_pose_recovery?: string;
    cameras?: string;
    alignment_diagnostics?: string;
    fusion_diagnostics?: string;
    visibility_graph?: string;
    consistency_diagnostics?: string;
    mesh?: string;
    mesh_diagnostics?: string;
    scene_splat?: string;
    gaussian_raw_model?: string;
    gaussian_model?: string;
    gaussian_training_result?: string;
    gaussian_progress?: string;
    gaussian_dataset?: string;
    gaussian_replay_dataset?: string;
    gaussian_replay_record?: string;
    gaussian_evaluation?: string;
    gaussian_test_evaluation?: string;
    gaussian_test_decision?: string;
    gaussian_export_metadata?: string;
    gaussian_canonical?: string;
    gaussian_camera_path?: string;
    gaussian_bundle?: string;
    vggt_ba_diagnostics?: string;
    vggt_ba_window_graph?: string;
    vggt_ba_initialization_diagnostics?: string;
    gaussian_vggt_filtered_model?: string;
    gaussian_vggt_filter_diagnostics?: string;
    gaussian_vggt_filter_mask?: string;
    gaussian_vggt_filtered_evaluation?: string;
    gaussian_vggt_filtered_export_metadata?: string;
    gaussian_vggt_filtered_canonical?: string;
    scene_splat_vggt_filtered?: string;
    gaussian_vggt_filtered_bundle?: string;
    collision_mesh?: string;
    navigation?: string;
    navigation_diagnostics?: string;
    video_probe?: string;
    video_frame_selection?: string;
    video_keyframe_timing?: string;
    video_registration_diagnostics?: string;
    video_initial_registration_expansion?: string;
    video_registration_recovery?: string;
    colmap_timing?: string;
    video_keyframe_contact_sheet?: string;
    scene_graph?: string;
    log?: string;
  };
  sfm_feature_profile?: SfmFeatureProfile;
  sfm_feature_effective_profile?: SfmFeatureProfile;
  sfm_local_matcher?: SfmLocalMatcher;
  sfm_local_matcher_effective?: SfmLocalMatcher;
  sfm_pairing?: SfmPairing;
  sfm_pairing_effective?: SfmPairing;
  sfm_geometric_verification?: SfmGeometricVerification;
  sfm_geometric_verification_effective?: SfmGeometricVerification;
  sfm_camera_calibration?: SfmCameraCalibration;
  sfm_camera_calibration_effective?: SfmCameraCalibration;
  gaussian_geometry_source?: GaussianGeometrySource;
  gaussian_geometry_effective_source?: GaussianGeometrySource | null;
  gaussian_geometry_fallback_applied?: boolean;
  gaussian_geometry_fallback_reason?: string | null;
  gaussian_postprocess?: GaussianPostprocess;
  gaussian_postprocess_status?: "not_requested" | "pending" | "available" | "unavailable";
  gaussian_postprocess_reason?: string | null;
  gaussian_sor_filter?: GaussianSorFilter;
  gaussian_sor_filter_status?: "pending" | "disabled" | "available" | "unavailable";
  gaussian_sor_filter_reason?: string | null;
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
    video_profile?: string;
    video_duration_seconds?: number;
    video_orientation?: string;
    video_candidate_count?: number;
    video_initial_selected_count?: number;
    video_base_selected_count?: number;
    video_adaptive_selected_count?: number;
    video_recovery_selected_count?: number;
    video_selected_count?: number;
    video_registered_count?: number;
    video_registration_rate?: number;
    video_registration_temporal_coverage?: number;
    video_registration_recovery_status?: string;
    video_registration_recovery_rounds?: number;
    video_registration_recovery_registered_gain?: number;
    sfm_feature_profile?: SfmFeatureProfile;
    sfm_local_matcher_profile?: SfmLocalMatcher;
    sfm_local_matcher?: string;
    sfm_pairing?: SfmPairing | "sequential";
    sfm_geometric_verification_profile?: SfmGeometricVerification;
    sfm_camera_calibration_profile?: SfmCameraCalibration;
    sfm_camera_model?: string;
    sfm_camera_planned_count?: number;
    sfm_camera_initial_count?: number;
    sfm_camera_final_count?: number;
    sfm_camera_prior_focal_count?: number;
    sfm_camera_warning_count?: number;
    sfm_camera_median_focal_length_ratio?: number;
    sfm_pose_health_status?: string;
    sfm_effective_mapper?: string;
    sfm_pose_recovery_status?: string;
    sfm_pose_recovery_applied?: boolean;
    sfm_pose_recovery_removed_camera_count?: number;
    sfm_median_reprojection_error_pixels?: number;
    sfm_median_track_length?: number;
    sfm_view_graph_verified_edge_count?: number;
    sfm_view_graph_component_count?: number;
    sfm_view_graph_largest_component_ratio?: number;
    sfm_view_graph_isolated_node_count?: number;
    sfm_view_graph_guided_inlier_count?: number;
    gaussian_geometry_source?: GaussianGeometrySource;
    gaussian_geometry_effective_source?: GaussianGeometrySource;
    gaussian_geometry_fallback_applied?: boolean;
    gaussian_geometry_fallback_reason?: string;
    vggt_ba_profile?: string;
    vggt_ba_trajectory_status?: string;
    vggt_ba_verified_nonlocal_edge_count?: number;
    vggt_ba_supported_camera_count?: number;
    vggt_ba_point_count?: number;
    gaussian_postprocess?: GaussianPostprocess;
    gaussian_postprocess_status?: string;
    gaussian_postprocess_reason?: string;
    gaussian_vggt_filter_input_count?: number;
    gaussian_vggt_filter_kept_count?: number;
    gaussian_vggt_filter_removed_count?: number;
    gaussian_vggt_filtered_validation_psnr?: number;
    gaussian_vggt_filtered_validation_ssim?: number;
    gaussian_sor_filter_input_count?: number;
    gaussian_sor_filter_kept_count?: number;
    gaussian_sor_filter_removed_count?: number;
    gaussian_count?: number;
    sfm_diagnostics_status?: string;
    sfm_diagnostics_reason?: string;
    sfm_diagnostics_image_count?: number;
    sfm_diagnostics_registered_image_count?: number;
    sfm_diagnostics_keypoint_count?: number;
    sfm_diagnostics_pair_count?: number;
    sfm_diagnostics_match_count?: number;
    sfm_diagnostics_inlier_count?: number;
    sfm_diagnostics_bytes?: number;
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
  sfm_feature_profiles?: SfmFeatureStatus[];
  sfm_camera_calibrations?: SfmCameraCalibrationStatus[];
  gaussian_geometry_sources?: ExperimentalOptionStatus<GaussianGeometrySource>[];
  gaussian_postprocessors?: ExperimentalOptionStatus<GaussianPostprocess>[];
  video_ingestion?: {
    available: boolean;
    reason: string | null;
    supported_profiles: string[];
    max_duration_seconds: number;
    max_size_bytes: number;
    max_keyframes: number;
  };
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
  const [sfmFeatureProfile, setSfmFeatureProfile] =
    useState<SfmFeatureProfile>("sift_v1");
  const [sfmLocalMatcher, setSfmLocalMatcher] =
    useState<SfmLocalMatcher>("bruteforce");
  const [sfmPairing, setSfmPairing] = useState<SfmPairing>("exhaustive");
  const [sfmGeometricVerification, setSfmGeometricVerification] =
    useState<SfmGeometricVerification>("default_v1");
  const [sfmCameraCalibration, setSfmCameraCalibration] =
    useState<SfmCameraCalibration>("shared_simple_radial_v1");
  const [gaussianTrainer, setGaussianTrainer] = useState<GaussianTrainer>("graphdeco");
  const [gaussianGeometrySource, setGaussianGeometrySource] =
    useState<GaussianGeometrySource>("colmap");
  const [gaussianPostprocess, setGaussianPostprocess] =
    useState<GaussianPostprocess>("none");
  const [gaussianSorFilter, setGaussianSorFilter] =
    useState<GaussianSorFilter>("on");
  const [gaussianLongestEdge, setGaussianLongestEdge] = useState(1280);
  const [videoRotation, setVideoRotation] = useState<VideoRotation>("auto");
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
  const [gaussianVariant, setGaussianVariant] = useState<GaussianVariant>("original");
  const [viewerMode, setViewerMode] = useState<ViewerMode>("point_cloud");
  const [meshSettings, setMeshSettings] = useState<MeshSettings>(defaultMeshSettings);
  const [selectedMeshVariantId, setSelectedMeshVariantId] = useState<string | null>(null);
  const [isBuildingMesh, setIsBuildingMesh] = useState(false);
  const [isBuildingNavigation, setIsBuildingNavigation] = useState(false);
  const [isChangingLifecycle, setIsChangingLifecycle] = useState(false);
  const [backendStatuses, setBackendStatuses] = useState<Record<GeometryBackend, BackendStatus> | null>(null);
  const [viewerFocus, setViewerFocus] = useState(false);
  const [inspectionRequest, setInspectionRequest] = useState<{
    id: number;
    tab: SfmInspectionTab;
  } | null>(null);
  const [activeInspectionTab, setActiveInspectionTab] = useState<SfmInspectionTab | null>(null);
  const [error, setError] = useState<string | null>(null);

  const selectedMode = modeOptions.find((option) => option.id === mode) ?? modeOptions[0];
  const selectedBackendStatus = backendStatuses?.[geometryBackend];
  const sfmFeatureStatuses = selectedBackendStatus?.sfm_feature_profiles ?? [];
  const selectedSfmFeatureStatus = sfmFeatureStatuses.find(
    (option) => option.id === sfmFeatureProfile
  );
  const sfmLocalMatcherStatuses = selectedSfmFeatureStatus?.local_matchers ?? [];
  const selectedSfmLocalMatcherStatus = sfmLocalMatcherStatuses.find(
    (option) => option.id === sfmLocalMatcher
  );
  const sfmLocalMatcherNotice = selectedSfmLocalMatcherStatus?.reason
    ? selectedSfmLocalMatcherStatus
    : sfmLocalMatcherStatuses.find((option) => option.available === false);
  const sfmPairingStatuses = selectedSfmLocalMatcherStatus?.pairings ?? [];
  const selectedSfmPairingStatus = sfmPairingStatuses.find(
    (option) => option.id === sfmPairing
  );
  const selectedSfmPairingSupportsMode = isSfmPairingModeSupported(
    selectedSfmPairingStatus,
    mode
  );
  const sfmPairingNotice = selectedSfmPairingStatus?.reason
    ? selectedSfmPairingStatus
    : sfmPairingStatuses.find((option) => option.available === false);
  const sfmGeometricVerificationStatuses =
    selectedSfmPairingStatus?.geometric_verifications ?? [];
  const selectedSfmGeometricVerificationStatus =
    sfmGeometricVerificationStatuses.find(
      (option) => option.id === sfmGeometricVerification
    );
  const sfmGeometricVerificationNotice =
    selectedSfmGeometricVerificationStatus?.reason
      ? selectedSfmGeometricVerificationStatus
      : sfmGeometricVerificationStatuses.find(
          (option) => option.available === false
        );
  const sfmCameraCalibrationStatuses =
    selectedBackendStatus?.sfm_camera_calibrations ?? [];
  const selectedSfmCameraCalibrationStatus =
    sfmCameraCalibrationStatuses.find(
      (option) => option.id === sfmCameraCalibration
    );
  const sfmCameraCalibrationNotice =
    selectedSfmCameraCalibrationStatus?.reason
      ? selectedSfmCameraCalibrationStatus
      : sfmCameraCalibrationStatuses.find(
          (option) => option.available === false
        );
  const usesColmapFeatureStage = ["colmap", "colmap_vggt", "project_3dgs"].includes(
    geometryBackend
  );
  const gaussianTrainerStatuses = selectedBackendStatus?.gaussian_trainers ?? [];
  const selectedGaussianTrainerStatus = findGaussianTrainerStatus(
    gaussianTrainerStatuses,
    gaussianTrainer
  );
  const gaussianGeometryStatuses = selectedBackendStatus?.gaussian_geometry_sources ?? [];
  const gaussianPostprocessStatuses = selectedBackendStatus?.gaussian_postprocessors ?? [];
  const selectedGaussianGeometryStatus = gaussianGeometryStatuses.find(
    (option) => option.id === gaussianGeometrySource
  );
  const selectedGaussianPostprocessStatus = gaussianPostprocessStatuses.find(
    (option) => option.id === gaussianPostprocess
  );
  const selectedBackendAvailable = selectedBackendStatus?.available ?? true;
  const selectedVideoAvailable =
    mode !== "video" || (selectedBackendStatus?.video_ingestion?.available ?? true);
  const selectedOutputSupported =
    selectedBackendStatus?.supported_outputs.includes(outputType) ?? true;
  const hasProductPointCloud = Boolean(manifest?.assets.point_cloud);
  const selectedProductPointCloudAsset = hasProductPointCloud
    ? pointCloudVariant === "aligned" && manifest?.assets.point_cloud_aligned
      ? manifest.assets.point_cloud_aligned
      : manifest?.assets.point_cloud
    : null;
  const selectedPointCloudAsset =
    selectedProductPointCloudAsset ??
    (manifest?.assets.sfm_sparse_point_cloud
      ? pointCloudVariant === "aligned" && manifest?.assets.point_cloud_aligned
        ? manifest.assets.point_cloud_aligned
        : manifest.assets.sfm_sparse_point_cloud
      : null);
  const showingSfmSparsePointCloud = Boolean(
    manifest?.assets.sfm_sparse_point_cloud && !hasProductPointCloud
  );
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
  const sfmDiagnosticsUrl = useMemo(() => {
    if (!manifest?.assets.sfm_diagnostics) {
      return null;
    }
    return `/api/jobs/${manifest.job_id}/assets/${manifest.assets.sfm_diagnostics}`;
  }, [manifest]);
  const alignmentDiagnosticsUrl = useMemo(() => {
    if (!manifest?.assets.alignment_diagnostics) {
      return null;
    }
    return `/api/jobs/${manifest.job_id}/assets/${manifest.assets.alignment_diagnostics}`;
  }, [manifest]);
  const selectedSplatAsset =
    gaussianVariant === "vggt_filtered" && manifest?.assets.scene_splat_vggt_filtered
      ? manifest.assets.scene_splat_vggt_filtered
      : manifest?.assets.scene_splat;
  const selectedSplatMetadata =
    gaussianVariant === "vggt_filtered" && manifest?.assets.gaussian_vggt_filtered_export_metadata
      ? manifest.assets.gaussian_vggt_filtered_export_metadata
      : manifest?.assets.gaussian_export_metadata;
  const splatUrl = useMemo(() => {
    if (!manifest || !selectedSplatAsset) {
      return null;
    }
    return `/api/jobs/${manifest.job_id}/assets/${selectedSplatAsset}`;
  }, [manifest, selectedSplatAsset]);
  const splatMetadataUrl = useMemo(() => {
    if (!manifest || !selectedSplatMetadata) {
      return null;
    }
    return `/api/jobs/${manifest.job_id}/assets/${selectedSplatMetadata}`;
  }, [manifest, selectedSplatMetadata]);
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
  const hasPointCloud = Boolean(
    manifest?.assets.point_cloud ||
      manifest?.assets.point_cloud_aligned ||
      manifest?.assets.sfm_sparse_point_cloud
  );
  const hasMesh = Boolean(selectedMeshAsset);
  const hasSplat = Boolean(manifest?.assets.scene_splat);
  const viewerAsset =
    viewerMode === "point_cloud"
      ? selectedPointCloudAsset
      : viewerMode === "mesh"
        ? selectedMeshAsset
        : selectedSplatAsset;
  const visiblePointCloudUrl = viewerMode === "point_cloud" ? pointCloudUrl : null;
  const visibleMeshUrl = viewerMode === "mesh" ? meshUrl : null;
  const visibleSplatUrl = viewerMode === "gaussian_splat" ? splatUrl : null;
  const evidenceStages = useMemo(
    () =>
      buildEvidenceStages({
        hasDiagnostics: Boolean(manifest?.assets.sfm_diagnostics),
        hasSparseGeometry: Boolean(manifest?.assets.sfm_sparse_point_cloud),
        hasGaussian: Boolean(manifest?.assets.scene_splat),
        imageCount: manifest?.metrics.sfm_diagnostics_image_count,
        registeredImageCount: manifest?.metrics.sfm_diagnostics_registered_image_count,
        pairCount: manifest?.metrics.sfm_diagnostics_pair_count,
        sparsePointCount: manifest?.metrics.num_points,
        gaussianCount: manifest?.metrics.gaussian_count
      }),
    [manifest]
  );
  const activeEvidence: EvidenceStageId | null = activeInspectionTab
    ? activeInspectionTab === "matches"
      ? "matching"
      : "input"
    : viewerMode === "point_cloud"
      ? "sparse"
      : viewerMode === "gaussian_splat"
        ? "gaussian"
        : null;

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

  useEffect(() => {
    if (!viewerFocus) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setViewerFocus(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [viewerFocus]);

  useEffect(() => {
    setInspectionRequest(null);
    setActiveInspectionTab(null);
  }, [manifest?.job_id]);

  function selectViewerMode(mode: ViewerMode) {
    setInspectionRequest(null);
    setActiveInspectionTab(null);
    setViewerMode(mode);
  }

  function requestEvidence(stage: EvidenceStageId) {
    if (stage === "input" || stage === "matching") {
      setViewerMode("gaussian_splat");
      const tab: SfmInspectionTab = stage === "matching" ? "matches" : "nearest";
      setInspectionRequest((current) => ({ id: (current?.id ?? 0) + 1, tab }));
      return;
    }
    setInspectionRequest(null);
    setActiveInspectionTab(null);
    setViewerMode(stage === "sparse" ? "point_cloud" : "gaussian_splat");
  }

  function applyManifest(nextManifest: Manifest, selectNewestMeshVariant = false, preferPointCloud = false) {
    setManifest(nextManifest);
    setPointCloudVariant(nextManifest.assets.point_cloud_aligned ? "aligned" : "raw");
    setGaussianVariant(
      nextManifest.assets.scene_splat_vggt_filtered ? "vggt_filtered" : "original"
    );
    setViewerMode((current) => {
      const hasProductPointCloud = Boolean(
        nextManifest.assets.point_cloud || nextManifest.assets.point_cloud_aligned
      );
      const hasPointCloud = Boolean(
        hasProductPointCloud || nextManifest.assets.sfm_sparse_point_cloud
      );
      const hasMesh = Boolean(nextManifest.assets.mesh);
      const hasSplat = Boolean(nextManifest.assets.scene_splat);
      if (
        nextManifest.output_type === "gaussian_splat" &&
        hasSplat &&
        (preferPointCloud || !manifest?.assets.scene_splat)
      ) {
        return "gaussian_splat";
      }
      if (preferPointCloud && hasProductPointCloud) {
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
    if (nextMode === "video") {
      setGeometryBackend("project_3dgs");
      setOutputType("gaussian_splat");
      setSfmCameraCalibration(
        defaultSfmCameraCalibration("project_3dgs")
      );
    } else if (gaussianGeometrySource === "vggt_ba") {
      setGaussianGeometrySource("colmap");
    }
    setFiles([]);
    setError(null);
  }

  function onGeometryBackendChange(nextBackend: GeometryBackend) {
    setGeometryBackend(nextBackend);
    setSfmCameraCalibration(defaultSfmCameraCalibration(nextBackend));
    setError(null);
  }

  function onFilesChange(event: ChangeEvent<HTMLInputElement>) {
    setFiles(Array.from(event.target.files ?? []));
    setError(null);
  }

  async function createJob() {
    if (mode === "video" && (geometryBackend !== "project_3dgs" || outputType !== "gaussian_splat")) {
      setError("视频模式目前仅支持 Project 3DGS + Gaussian Splat。");
      return;
    }
    if (mode === "video" && selectedBackendStatus?.video_ingestion?.available === false) {
      setError(selectedBackendStatus.video_ingestion.reason ?? "服务器缺少视频处理工具。");
      return;
    }
    if (
      usesColmapFeatureStage &&
      sfmFeatureProfile !== "sift_v1" &&
      selectedSfmFeatureStatus?.available !== true
    ) {
      setError(
        selectedSfmFeatureStatus?.reason ?? "服务器缺少所选 COLMAP 特征模型。"
      );
      return;
    }
    if (
      usesColmapFeatureStage &&
      (
        selectedSfmLocalMatcherStatus?.available === false ||
        (sfmLocalMatcher !== "bruteforce" &&
          selectedSfmLocalMatcherStatus?.available !== true)
      )
    ) {
      setError(
        selectedSfmLocalMatcherStatus?.reason ??
          "服务器缺少所选 COLMAP 局部匹配模型。"
      );
      return;
    }
    if (
      usesColmapFeatureStage &&
      !isSfmPairingAvailable(sfmPairing, selectedSfmPairingStatus, mode)
    ) {
      setError(
        selectedSfmPairingStatus?.reason ??
          "所选图像对策略不支持当前输入模式或服务器配置。"
      );
      return;
    }
    if (
      usesColmapFeatureStage &&
      !isSfmGeometricVerificationAvailable(
        sfmGeometricVerification,
        selectedSfmGeometricVerificationStatus
      )
    ) {
      setError(
        selectedSfmGeometricVerificationStatus?.reason ??
          "服务器不支持所选两视图几何验证配置。"
      );
      return;
    }
    if (
      usesColmapFeatureStage &&
      !isSfmCameraCalibrationAvailable(
        sfmCameraCalibration,
        selectedSfmCameraCalibrationStatus,
        mode,
        geometryBackend
      )
    ) {
      setError(
        selectedSfmCameraCalibrationStatus?.reason ??
          "服务器或当前输入模式不支持所选相机标定配置。"
      );
      return;
    }
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
    if (geometryBackend === "project_3dgs" && selectedGaussianGeometryStatus?.available === false) {
      setError(selectedGaussianGeometryStatus.reason ?? "所选高斯几何来源不可用。");
      return;
    }
    if (gaussianGeometrySource === "vggt_ba" && mode !== "video") {
      setError("VGGT + BA 几何首版仅支持视频模式。");
      return;
    }
    if (geometryBackend === "project_3dgs" && selectedGaussianPostprocessStatus?.available === false) {
      setError(selectedGaussianPostprocessStatus.reason ?? "所选高斯后处理不可用。");
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
      if (usesColmapFeatureStage) {
        form.append("sfm_feature_profile", sfmFeatureProfile);
        form.append("sfm_local_matcher", sfmLocalMatcher);
        form.append("sfm_pairing", sfmPairing);
        form.append("sfm_geometric_verification", sfmGeometricVerification);
        form.append("sfm_camera_calibration", sfmCameraCalibration);
      }
      if (geometryBackend === "project_3dgs") {
        form.append("gaussian_trainer", gaussianTrainer);
        form.append("gaussian_geometry_source", gaussianGeometrySource);
        form.append("gaussian_postprocess", gaussianPostprocess);
        form.append("gaussian_sor_filter", gaussianSorFilter);
        form.append("gaussian_longest_edge", String(gaussianLongestEdge));
      }
      if (mode === "video") {
        form.append("video_keyframe_profile", "standard_v2");
        form.append("video_rotation", videoRotation);
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
    <main className={viewerFocus ? "app-shell viewer-focus" : "app-shell"}>
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
                onChange={(event) =>
                  onGeometryBackendChange(
                    event.target.value as GeometryBackend
                  )
                }
              >
                {backendOptions.map((option) => (
                  <option
                    disabled={
                      !isBackendAvailable(option.id, backendStatuses) ||
                      (mode === "video" && option.id !== "project_3dgs")
                    }
                    key={option.id}
                    value={option.id}
                  >
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
                  <option
                    disabled={
                      !isOutputSupported(option.id, selectedBackendStatus) ||
                      (mode === "video" && option.id !== "gaussian_splat")
                    }
                    key={option.id}
                    value={option.id}
                  >
                    {isOutputSupported(option.id, selectedBackendStatus) ? option.label : `${option.label}（不可用）`}
                  </option>
                ))}
              </select>
            </label>

            {usesColmapFeatureStage && (
              <>
                <label>
                  <span>特征提取</span>
                  <select
                    value={sfmFeatureProfile}
                    onChange={(event) =>
                      setSfmFeatureProfile(event.target.value as SfmFeatureProfile)
                    }
                  >
                    {sfmFeatureOptions.map((option) => {
                      const status = sfmFeatureStatuses.find(
                        (candidate) => candidate.id === option.id
                      );
                      const unavailable =
                        option.id !== "sift_v1" && status?.available !== true;
                      return (
                        <option
                          disabled={unavailable}
                          key={option.id}
                          value={option.id}
                        >
                          {unavailable
                            ? `${option.label}（不可用）`
                            : option.label}
                        </option>
                      );
                    })}
                  </select>
                  <small>
                    每个 Job 只使用一种 COLMAP 特征；局部匹配、图像对、几何验证和 Mapper 分别控制。
                  </small>
                  {selectedSfmFeatureStatus?.reason && (
                    <small>{selectedSfmFeatureStatus.reason}</small>
                  )}
                  {selectedSfmFeatureStatus?.setup_command && (
                    <small>{selectedSfmFeatureStatus.setup_command}</small>
                  )}
                </label>

                <label>
                  <span>局部匹配</span>
                  <select
                    value={sfmLocalMatcher}
                    onChange={(event) =>
                      setSfmLocalMatcher(event.target.value as SfmLocalMatcher)
                    }
                  >
                    {sfmLocalMatcherOptions.map((option) => {
                      const status = sfmLocalMatcherStatuses.find(
                        (candidate) => candidate.id === option.id
                      );
                      const unavailable =
                        status?.available === false ||
                        (option.id !== "bruteforce" && status?.available !== true);
                      return (
                        <option
                          disabled={unavailable}
                          key={option.id}
                          value={option.id}
                        >
                          {unavailable
                            ? `${option.label}（不可用）`
                            : option.label}
                        </option>
                      );
                    })}
                  </select>
                  <small>
                    只改变已选图像对内的 descriptor 匹配；LightGlue 固定最小分数 0.1。
                  </small>
                  {sfmLocalMatcherNotice?.reason && (
                    <small>{sfmLocalMatcherNotice.reason}</small>
                  )}
                  {sfmLocalMatcherNotice?.setup_command && (
                    <small>{sfmLocalMatcherNotice.setup_command}</small>
                  )}
                </label>

                <label>
                  <span>图像对策略</span>
                  <select
                    value={sfmPairing}
                    onChange={(event) =>
                      setSfmPairing(event.target.value as SfmPairing)
                    }
                  >
                    {sfmPairingOptions.map((option) => {
                      const status = sfmPairingStatuses.find(
                        (candidate) => candidate.id === option.id
                      );
                      const unavailable = !isSfmPairingAvailable(
                        option.id,
                        status,
                        mode
                      );
                      return (
                        <option
                          disabled={unavailable}
                          key={option.id}
                          value={option.id}
                        >
                          {unavailable
                            ? `${option.label}（不可用）`
                            : option.label}
                        </option>
                      );
                    })}
                  </select>
                  <small>
                    决定哪些图像对进入局部匹配；Sequential + Loop 仅用于视频，Vocab Tree 仅用于多图。
                  </small>
                  {selectedSfmPairingStatus !== undefined &&
                    !selectedSfmPairingSupportsMode && (
                      <small>所选图像对策略不支持当前输入模式。</small>
                    )}
                  {sfmPairingNotice?.reason && (
                    <small>{sfmPairingNotice.reason}</small>
                  )}
                  {sfmPairingNotice?.setup_command && (
                    <small>{sfmPairingNotice.setup_command}</small>
                  )}
                </label>

                <label>
                  <span>两视图几何验证</span>
                  <select
                    value={sfmGeometricVerification}
                    onChange={(event) =>
                      setSfmGeometricVerification(
                        event.target.value as SfmGeometricVerification
                      )
                    }
                  >
                    {sfmGeometricVerificationOptions.map((option) => {
                      const status = sfmGeometricVerificationStatuses.find(
                        (candidate) => candidate.id === option.id
                      );
                      const unavailable =
                        !isSfmGeometricVerificationAvailable(option.id, status);
                      return (
                        <option
                          disabled={unavailable}
                          key={option.id}
                          value={option.id}
                        >
                          {unavailable
                            ? `${option.label}（不可用）`
                            : option.label}
                        </option>
                      );
                    })}
                  </select>
                  <small>
                    Guided 只扩展满足已估计几何的对应；RANSAC 原始参数保持同一 COLMAP build 默认值。
                  </small>
                  {sfmGeometricVerificationNotice?.reason && (
                    <small>{sfmGeometricVerificationNotice.reason}</small>
                  )}
                  {sfmGeometricVerificationNotice?.setup_command && (
                    <small>{sfmGeometricVerificationNotice.setup_command}</small>
                  )}
                </label>

                <label>
                  <span>相机标定</span>
                  <select
                    value={sfmCameraCalibration}
                    onChange={(event) =>
                      setSfmCameraCalibration(
                        event.target.value as SfmCameraCalibration
                      )
                    }
                  >
                    {sfmCameraCalibrationOptions.map((option) => {
                      const status = sfmCameraCalibrationStatuses.find(
                        (candidate) => candidate.id === option.id
                      );
                      const unavailable = !isSfmCameraCalibrationAvailable(
                        option.id,
                        status,
                        mode,
                        geometryBackend
                      );
                      const label = status?.is_default
                        ? `${option.label}（默认）`
                        : status?.experimental
                          ? `${option.label}（实验）`
                          : option.label;
                      return (
                        <option
                          disabled={unavailable}
                          key={option.id}
                          value={option.id}
                        >
                          {unavailable ? `${label}（不可用）` : label}
                        </option>
                      );
                    })}
                  </select>
                  <small>
                    Shared 固定一组内参；Auto-grouped 按设备、镜头/焦距、尺寸和方向分组，证据不足时每图独立。
                  </small>
                  {sfmCameraCalibrationNotice?.reason && (
                    <small>{sfmCameraCalibrationNotice.reason}</small>
                  )}
                  {sfmCameraCalibrationNotice?.setup_command && (
                    <small>{sfmCameraCalibrationNotice.setup_command}</small>
                  )}
                </label>
              </>
            )}

            {geometryBackend === "project_3dgs" && (
              <>
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
                          },
                          {
                            id: "mcmc" as const,
                            label: "MCMC v1（实验，gsplat）",
                            available: true,
                            reason: null,
                            setup_command: null,
                            revision: "gsplat-1.5.3-mcmc-v1",
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
                      v7 配置：30,000 次迭代上限 · 自动使用全部可见 GPU · 3NN（三近邻）初始化 ·
                      关闭屏幕半径剪枝 · 由验证集选择模型 · 归一化任意单位。
                    </small>
                  )}
                  {gaussianTrainer === "mcmc" && (
                    <small>
                      实验性 MCMC v1：30,000 次迭代 · 全局最多 3,000,000 个高斯 ·
                      自动使用全部可见 GPU · 冻结 3NN 初始化 · 由验证集选择模型 · 归一化任意单位。
                    </small>
                  )}
                </label>
                <label>
                  <span>几何来源</span>
                  <select
                    value={gaussianGeometrySource}
                    onChange={(event) =>
                      setGaussianGeometrySource(event.target.value as GaussianGeometrySource)
                    }
                  >
                    {(gaussianGeometryStatuses.length > 0
                      ? gaussianGeometryStatuses
                      : [
                          {
                            id: "colmap" as const,
                            label: "COLMAP",
                            available: true,
                            reason: null,
                            experimental: false
                          },
                          {
                            id: "vggt_ba" as const,
                            label: "VGGT + BA",
                            available: false,
                            reason: "服务器尚未报告 VGGT-BA 能力",
                            experimental: true,
                            supported_modes: ["video" as Mode]
                          }
                        ]
                    ).map((option) => (
                      <option
                        key={option.id}
                        value={option.id}
                        disabled={!option.available || (option.id === "vggt_ba" && mode !== "video")}
                      >
                        {option.label}
                        {option.experimental ? "（实验）" : "（默认）"}
                        {!option.available ? "（不可用）" : ""}
                      </option>
                    ))}
                  </select>
                  {selectedGaussianGeometryStatus?.reason && (
                    <small>{selectedGaussianGeometryStatus.reason}</small>
                  )}
                  {selectedGaussianGeometryStatus?.available === false &&
                    selectedGaussianGeometryStatus.setup_command && (
                      <small>
                        安装：<code>{selectedGaussianGeometryStatus.setup_command}</code>
                      </small>
                    )}
                  {gaussianGeometrySource === "vggt_ba" && (
                    <small>
                      研究实验：8/4 重叠窗口 · 弱帧恢复 · ALIKED/VGGSfM 局部 BA · 全局 Sim(3) 图 ·
                      COLMAP 补注册/global BA。仅在有界恢复后的几何质量门失败时显式回退 COLMAP；结果会标明实际来源。
                    </small>
                  )}
                </label>
                <label>
                  <span>训练后清理</span>
                  <select
                    value={gaussianPostprocess}
                    onChange={(event) =>
                      setGaussianPostprocess(event.target.value as GaussianPostprocess)
                    }
                  >
                    {(gaussianPostprocessStatuses.length > 0
                      ? gaussianPostprocessStatuses
                      : [
                          {
                            id: "none" as const,
                            label: "关闭",
                            available: true,
                            reason: null,
                            experimental: false
                          },
                          {
                            id: "vggt_visibility_v1" as const,
                            label: "VGGT Train-depth 清理",
                            available: false,
                            reason: "服务器尚未报告 VGGT 后处理能力",
                            experimental: true
                          }
                        ]
                    ).map((option) => (
                      <option disabled={!option.available} key={option.id} value={option.id}>
                        {option.label}
                        {option.experimental ? "（实验）" : "（默认）"}
                        {!option.available ? "（不可用）" : ""}
                      </option>
                    ))}
                  </select>
                  {selectedGaussianPostprocessStatus?.reason && (
                    <small>{selectedGaussianPostprocessStatus.reason}</small>
                  )}
                  {selectedGaussianPostprocessStatus?.available === false &&
                    selectedGaussianPostprocessStatus.setup_command && (
                      <small>
                        安装：<code>{selectedGaussianPostprocessStatus.setup_command}</code>
                      </small>
                    )}
                  {gaussianPostprocess === "vggt_visibility_v1" && (
                    <small>
                      只使用 Train 图像的 VGGT depth 删除有多视图自由空间矛盾的漂浮物和拍摄包络外的大型高斯；保留原始结果供 A/B，不补墙、不使用 Test。
                    </small>
                  )}
                </label>
                <label>
                  <span>SOR 浮点清理</span>
                  <select
                    value={gaussianSorFilter}
                    onChange={(event) =>
                      setGaussianSorFilter(event.target.value as GaussianSorFilter)
                    }
                  >
                    <option value="on">开启（默认）</option>
                    <option value="off">关闭</option>
                  </select>
                  <small>
                    导出前用保守 SOR band 参数原位删除孤立低透明度高斯（渲染无损证据）；失败自动回退为未过滤结果。
                  </small>
                </label>
                <label>
                  <span>训练图像最长边</span>
                  <select
                    value={gaussianLongestEdge}
                    onChange={(event) => setGaussianLongestEdge(Number(event.target.value))}
                  >
                    <option value={1280}>1280px（默认）</option>
                    <option value={1920}>1920px</option>
                    <option value={2560}>2560px</option>
                    <option value={3072}>3072px</option>
                  </select>
                  <small>同时控制 COLMAP 去畸变训练图和 3DGS 训练/验证视图；分辨率越高，显存占用越大。</small>
                </label>
                {mode === "video" && (
                  <label>
                    <span>视频方向</span>
                    <select
                      value={videoRotation}
                      onChange={(event) => setVideoRotation(event.target.value as VideoRotation)}
                    >
                      <option value="auto">自动读取视频方向</option>
                      <option value="clockwise_90">强制顺时针旋转 90°</option>
                      <option value="counterclockwise_90">强制逆时针旋转 90°</option>
                      <option value="180">强制旋转 180°</option>
                    </select>
                    <small>
                      支持 10 秒–10 分钟、最大 2 GiB 的 MP4/MOV/M4V/WebM；覆盖 1080p30
                      竖拍输入，最多选择 800 张关键帧。坐标仍为归一化任意单位。
                    </small>
                    {selectedBackendStatus?.video_ingestion?.reason && (
                      <small>{selectedBackendStatus.video_ingestion.reason}</small>
                    )}
                  </label>
                )}
              </>
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
                accept={mode === "video" ? ".mp4,.mov,.m4v,.webm,video/*" : "image/*"}
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
            disabled={
              isSubmitting ||
              !selectedBackendAvailable ||
              !selectedVideoAvailable ||
              !selectedOutputSupported ||
              (geometryBackend === "project_3dgs" &&
                (selectedGaussianGeometryStatus?.available === false ||
                  selectedGaussianPostprocessStatus?.available === false))
            }
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
                    <button className={viewerMode === "point_cloud" ? "active" : ""} type="button" onClick={() => selectViewerMode("point_cloud")}>
                      {showingSfmSparsePointCloud ? "SfM 稀疏点云" : "点云（Point Cloud）"}
                    </button>
                  )}
                  {hasMesh && (
                    <button className={viewerMode === "mesh" ? "active" : ""} type="button" onClick={() => selectViewerMode("mesh")}>
                      网格（Mesh）
                    </button>
                  )}
                  {hasSplat && (
                    <button
                      className={viewerMode === "gaussian_splat" ? "active" : ""}
                      type="button"
                      onClick={() => selectViewerMode("gaussian_splat")}
                    >
                      高斯泼溅（Gaussian Splat）
                    </button>
                  )}
                </div>
              )}
              {hasPointCloud && viewerMode === "point_cloud" && (!showingSfmSparsePointCloud || hasAlignedPointCloud) && (
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
              {hasSplat && viewerMode === "gaussian_splat" && (
                <div className="variant-toggle" role="group" aria-label="高斯清理前后视图">
                  <button
                    className={gaussianVariant === "original" ? "active" : ""}
                    type="button"
                    onClick={() => setGaussianVariant("original")}
                  >
                    Original
                  </button>
                  <button
                    className={gaussianVariant === "vggt_filtered" ? "active" : ""}
                    type="button"
                    onClick={() => setGaussianVariant("vggt_filtered")}
                    disabled={!manifest?.assets.scene_splat_vggt_filtered}
                  >
                    VGGT-filtered
                  </button>
                </div>
              )}
              <button
                className="icon-button"
                type="button"
                onClick={() => setViewerFocus((current) => !current)}
                disabled={!manifest}
                title={viewerFocus ? "退出专注查看（Esc）" : "隐藏两侧面板，扩大诊断工作区"}
              >
                {viewerFocus ? <Minimize2 size={17} aria-hidden="true" /> : <Maximize2 size={17} aria-hidden="true" />}
                <span>{viewerFocus ? "退出专注" : "专注查看"}</span>
              </button>
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
          {manifest && (
            <ReconstructionEvidenceRail
              active={activeEvidence}
              onSelect={requestEvidence}
              stages={evidenceStages}
            />
          )}
          <GeometryViewer
            pointCloudUrl={visiblePointCloudUrl}
            camerasUrl={viewerMode === "point_cloud" ? camerasUrl : null}
            alignmentDiagnosticsUrl={alignmentDiagnosticsUrl}
            pointCloudVariant={pointCloudVariant}
            meshUrl={visibleMeshUrl}
            splatUrl={visibleSplatUrl}
            splatMetadataUrl={viewerMode === "gaussian_splat" ? splatMetadataUrl : null}
            splatCameraPathUrl={viewerMode === "gaussian_splat" ? splatCameraPathUrl : null}
            jobId={manifest?.job_id ?? null}
            sfmDiagnosticsUrl={viewerMode === "gaussian_splat" ? sfmDiagnosticsUrl : null}
            inspectionRequest={inspectionRequest}
            onInspectionStateChange={setActiveInspectionTab}
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

          <dl className="metrics-grid core-metrics-grid">
            <div><dt>状态</dt><dd>{formatStatus(currentStatus?.status)}</dd></div>
            <div><dt>处理阶段</dt><dd>{formatStage(currentStatus?.stage)}</dd></div>
            <div><dt>注册率</dt><dd>{formatPercentMetric(currentStatus?.metrics.video_registration_rate)}</dd></div>
            <div><dt>时间覆盖</dt><dd>{formatPercentMetric(currentStatus?.metrics.video_registration_temporal_coverage)}</dd></div>
            <div><dt>候选 / 关键帧</dt><dd>{currentStatus?.metrics.video_candidate_count === undefined ? "-" : `${currentStatus.metrics.video_candidate_count} / ${currentStatus.metrics.video_selected_count ?? "-"}`}</dd></div>
            <div><dt>稀疏点</dt><dd>{formatInteger(currentStatus?.metrics.num_points)}</dd></div>
            <div><dt>SfM 诊断</dt><dd>{formatStatus(currentStatus?.metrics.sfm_diagnostics_status)}</dd></div>
            <div><dt>位姿健康</dt><dd>{formatStatus(currentStatus?.metrics.sfm_pose_health_status)}</dd></div>
            <div><dt>有效求解器</dt><dd>{formatPolicy(currentStatus?.metrics.sfm_effective_mapper)}</dd></div>
            <div><dt>位姿恢复</dt><dd>{formatStatus(currentStatus?.metrics.sfm_pose_recovery_status)}</dd></div>
            <div><dt>已注册图片</dt><dd>{formatRatio(currentStatus?.metrics.sfm_diagnostics_registered_image_count, currentStatus?.metrics.sfm_diagnostics_image_count)}</dd></div>
            <div><dt>匹配内点</dt><dd>{formatInteger(currentStatus?.metrics.sfm_diagnostics_inlier_count)}</dd></div>
            <div><dt>高斯数量</dt><dd>{formatInteger(currentStatus?.metrics.gaussian_count)}</dd></div>
            <div><dt>空间对齐</dt><dd>{formatPolicy(currentStatus?.metrics.alignment_status)}</dd></div>
            <div><dt>Validation PSNR</dt><dd>{currentStatus?.metrics.gaussian_vggt_filtered_validation_psnr === undefined ? "-" : `${currentStatus.metrics.gaussian_vggt_filtered_validation_psnr.toFixed(3)} dB`}</dd></div>
          </dl>

          <details className="result-details">
            <summary>完整运行指标</summary>
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
            {manifest && ["colmap", "colmap_vggt", "project_3dgs"].includes(manifest.geometry_backend) && (
              <div>
                <dt>SfM 特征</dt>
                <dd>
                  {formatSfmFeatureProfile(
                    manifest.sfm_feature_effective_profile ??
                      manifest.sfm_feature_profile ??
                      currentStatus?.metrics.sfm_feature_profile
                  )}
                </dd>
              </div>
            )}
            {manifest && ["colmap", "colmap_vggt", "project_3dgs"].includes(manifest.geometry_backend) && (
              <div>
                <dt>SfM 局部匹配</dt>
                <dd>
                  {formatSfmLocalMatcher(
                    manifest.sfm_local_matcher_effective ??
                      manifest.sfm_local_matcher ??
                      currentStatus?.metrics.sfm_local_matcher_profile
                  )}
                </dd>
              </div>
            )}
            {manifest && ["colmap", "colmap_vggt", "project_3dgs"].includes(manifest.geometry_backend) && (
              <div>
                <dt>SfM 图像对策略</dt>
                <dd>
                  {formatSfmPairing(
                    manifest.sfm_pairing_effective ??
                      manifest.sfm_pairing ??
                      currentStatus?.metrics.sfm_pairing
                  )}
                </dd>
              </div>
            )}
            {manifest && ["colmap", "colmap_vggt", "project_3dgs"].includes(manifest.geometry_backend) && (
              <div>
                <dt>SfM 几何验证</dt>
                <dd>
                  {formatSfmGeometricVerification(
                    manifest.sfm_geometric_verification_effective ??
                      manifest.sfm_geometric_verification ??
                      currentStatus?.metrics.sfm_geometric_verification_profile
                  )}
                </dd>
              </div>
            )}
            {manifest && ["colmap", "colmap_vggt", "project_3dgs"].includes(manifest.geometry_backend) && (
              <div>
                <dt>SfM 相机标定</dt>
                <dd>
                  {formatSfmCameraCalibration(
                    manifest.sfm_camera_calibration_effective ??
                      manifest.sfm_camera_calibration ??
                      currentStatus?.metrics.sfm_camera_calibration_profile,
                    manifest.geometry_backend
                  )}
                </dd>
              </div>
            )}
            {manifest && ["colmap", "colmap_vggt", "project_3dgs"].includes(manifest.geometry_backend) && (
              <div>
                <dt>SfM 相机组 / 警告</dt>
                <dd>
                  {currentStatus?.metrics.sfm_camera_final_count ?? "-"} / {currentStatus?.metrics.sfm_camera_warning_count ?? "-"}
                </dd>
              </div>
            )}
            {manifest?.geometry_backend === "project_3dgs" && (
              <div>
                <dt>SfM 位姿健康{currentStatus?.metrics.sfm_effective_mapper ? " / 求解器" : ""}</dt>
                <dd>
                  {formatStatus(currentStatus?.metrics.sfm_pose_health_status)}
                  {currentStatus?.metrics.sfm_effective_mapper
                    ? ` / ${formatPolicy(currentStatus.metrics.sfm_effective_mapper)}`
                    : ""}
                </dd>
              </div>
            )}
            {manifest?.geometry_backend === "project_3dgs" && currentStatus?.metrics.sfm_pose_recovery_status !== undefined && (
              <div>
                <dt>SfM 有界恢复</dt>
                <dd>
                  {formatStatus(currentStatus?.metrics.sfm_pose_recovery_status)}
                  {currentStatus?.metrics.sfm_pose_recovery_removed_camera_count
                    ? ` · 删除 ${currentStatus.metrics.sfm_pose_recovery_removed_camera_count} 个异常相机`
                    : ""}
                </dd>
              </div>
            )}
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
              <dt>请求的高斯几何</dt>
              <dd>{formatPolicy(manifest?.gaussian_geometry_source ?? currentStatus?.metrics.gaussian_geometry_source)}</dd>
            </div>
            <div>
              <dt>实际高斯几何</dt>
              <dd>
                {formatPolicy(
                  manifest?.gaussian_geometry_effective_source ??
                    currentStatus?.metrics.gaussian_geometry_effective_source
                )}
              </dd>
            </div>
            <div>
              <dt>VGGT-BA 轨迹</dt>
              <dd>{formatPolicy(currentStatus?.metrics.vggt_ba_trajectory_status)}</dd>
            </div>
            <div>
              <dt>验证闭环边</dt>
              <dd>{currentStatus?.metrics.vggt_ba_verified_nonlocal_edge_count ?? "-"}</dd>
            </div>
            <div>
              <dt>VGGT 后处理</dt>
              <dd>{formatStatus(manifest?.gaussian_postprocess_status ?? currentStatus?.metrics.gaussian_postprocess_status)}</dd>
            </div>
            {(manifest?.gaussian_sor_filter !== undefined ||
              currentStatus?.metrics.gaussian_sor_filter_removed_count !== undefined) && (
              <div>
                <dt>SOR 浮点清理</dt>
                <dd>
                  {formatStatus(manifest?.gaussian_sor_filter_status)}
                  {currentStatus?.metrics.gaussian_sor_filter_removed_count !== undefined &&
                    ` · 删除 ${currentStatus.metrics.gaussian_sor_filter_removed_count}`}
                </dd>
              </div>
            )}
            <div>
              <dt>高斯保留 / 删除</dt>
              <dd>
                {currentStatus?.metrics.gaussian_vggt_filter_kept_count === undefined
                  ? "-"
                  : `${currentStatus.metrics.gaussian_vggt_filter_kept_count} / ${currentStatus.metrics.gaussian_vggt_filter_removed_count ?? "-"}`}
              </dd>
            </div>
            <div>
              <dt>过滤版 Validation</dt>
              <dd>
                {currentStatus?.metrics.gaussian_vggt_filtered_validation_psnr === undefined
                  ? "-"
                  : `${currentStatus.metrics.gaussian_vggt_filtered_validation_psnr.toFixed(3)} dB · SSIM ${(currentStatus.metrics.gaussian_vggt_filtered_validation_ssim ?? 0).toFixed(4)}`}
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
              <dt>视频时长</dt>
              <dd>
                {currentStatus?.metrics.video_duration_seconds === undefined
                  ? "-"
                  : `${currentStatus.metrics.video_duration_seconds.toFixed(1)} 秒`}
              </dd>
            </div>
            <div>
              <dt>候选 / 关键帧</dt>
              <dd>
                {currentStatus?.metrics.video_candidate_count === undefined
                  ? "-"
                  : `${currentStatus.metrics.video_candidate_count} / ${currentStatus.metrics.video_selected_count ?? "-"}`}
              </dd>
            </div>
            <div>
              <dt>视频注册率</dt>
              <dd>
                {currentStatus?.metrics.video_registration_rate === undefined
                  ? "-"
                  : `${(currentStatus.metrics.video_registration_rate * 100).toFixed(1)}%`}
              </dd>
            </div>
            <div>
              <dt>注册时间覆盖</dt>
              <dd>
                {currentStatus?.metrics.video_registration_temporal_coverage === undefined
                  ? "-"
                  : `${(currentStatus.metrics.video_registration_temporal_coverage * 100).toFixed(1)}%`}
              </dd>
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
          </details>

          {(manifest?.gaussian_geometry_fallback_applied ??
            currentStatus?.metrics.gaussian_geometry_fallback_applied) && (
            <p className="error-message">
              VGGT-BA 已在完成有界弱帧恢复和注册质量门后回退到普通 COLMAP。
              实际几何来源为 COLMAP；此 Job 可查看，但不能计为成功的 VGGT-BA A/B 证据。
              原因：
              {formatPolicy(
                manifest?.gaussian_geometry_fallback_reason ??
                  currentStatus?.metrics.gaussian_geometry_fallback_reason
              )}
              。
            </p>
          )}
          {!(manifest?.gaussian_geometry_fallback_applied ??
            currentStatus?.metrics.gaussian_geometry_fallback_applied) &&
            currentStatus?.metrics.vggt_ba_trajectory_status === "open_trajectory_unverified" && (
            <p className="error-message">
              VGGT-BA 未找到通过几何验证的非局部闭环；当前结果是未验证开放轨迹，不能解释为无漂移全屋重建。
            </p>
          )}
          {manifest?.gaussian_postprocess_status === "unavailable" && (
            <p className="error-message">
              VGGT 后处理不可用：{manifest.gaussian_postprocess_reason ?? "未生成过滤版；原始 Gaussian 结果仍然有效。"}
            </p>
          )}

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
            <details className="result-details">
              <summary>结果资产</summary>
              <div className="asset-links">
              <AssetLink manifest={manifest} assetKey="sfm_diagnostics" label="SfM 前端诊断" />
              <AssetLink
                manifest={manifest}
                assetKey="sfm_camera_calibration_diagnostics"
                label="SfM 相机标定诊断"
              />
              <AssetLink manifest={manifest} assetKey="sfm_pose_health" label="SfM 位姿健康" />
              <AssetLink manifest={manifest} assetKey="sfm_pose_recovery" label="SfM 位姿恢复" />
              <AssetLink manifest={manifest} assetKey="sfm_sparse_point_cloud" label="SfM 稀疏点云" />
              <AssetLink manifest={manifest} assetKey="cameras" label="SfM 相机位姿" />
              <AssetLink manifest={manifest} assetKey="point_cloud" label="点云（Point Cloud）" />
              <AssetLink manifest={manifest} assetKey="point_cloud_aligned" label="对齐点云" />
              <AssetLink manifest={manifest} assetKey="mesh" label="三维网格（Mesh）" />
              <AssetLink manifest={manifest} assetKey="mesh_diagnostics" label="网格诊断" />
              <AssetLink manifest={manifest} assetKey="scene_splat" label="高斯浏览资产" />
              <AssetLink manifest={manifest} assetKey="gaussian_raw_model" label="训练器原始高斯模型" />
              <AssetLink manifest={manifest} assetKey="gaussian_replay_dataset" label="高斯重放数据约定" />
              <AssetLink manifest={manifest} assetKey="gaussian_replay_record" label="高斯重放记录" />
              <AssetLink manifest={manifest} assetKey="gaussian_canonical" label="标准高斯 PLY" />
              <AssetLink manifest={manifest} assetKey="gaussian_export_metadata" label="高斯导出元数据" />
              <AssetLink manifest={manifest} assetKey="gaussian_evaluation" label="高斯验证集评估" />
              <AssetLink manifest={manifest} assetKey="gaussian_test_evaluation" label="高斯留出测试集评估" />
              <AssetLink manifest={manifest} assetKey="gaussian_test_decision" label="高斯测试判定" />
              <AssetLink manifest={manifest} assetKey="gaussian_camera_path" label="高斯相机路径" />
              <AssetLink manifest={manifest} assetKey="gaussian_bundle" label="高斯结果包" />
              <AssetLink manifest={manifest} assetKey="vggt_ba_diagnostics" label="VGGT-BA 诊断" />
              <AssetLink manifest={manifest} assetKey="vggt_ba_window_graph" label="VGGT-BA 窗口图" />
              <AssetLink manifest={manifest} assetKey="vggt_ba_initialization_diagnostics" label="VGGT-BA Train 初始化诊断" />
              <AssetLink manifest={manifest} assetKey="scene_splat_vggt_filtered" label="VGGT 清理后高斯浏览资产" />
              <AssetLink manifest={manifest} assetKey="gaussian_vggt_filtered_canonical" label="VGGT 清理后标准高斯 PLY" />
              <AssetLink manifest={manifest} assetKey="gaussian_vggt_filter_diagnostics" label="VGGT 高斯清理诊断" />
              <AssetLink manifest={manifest} assetKey="gaussian_vggt_filter_mask" label="VGGT 高斯清理掩码" />
              <AssetLink manifest={manifest} assetKey="gaussian_vggt_filtered_evaluation" label="VGGT 清理后验证评估" />
              <AssetLink manifest={manifest} assetKey="gaussian_vggt_filtered_bundle" label="VGGT 清理后结果包" />
              <AssetLink manifest={manifest} assetKey="collision_mesh" label="碰撞网格" />
              <AssetLink manifest={manifest} assetKey="navigation" label="导航数据约定" />
              <AssetLink manifest={manifest} assetKey="navigation_diagnostics" label="导航诊断" />
              <AssetLink manifest={manifest} assetKey="video_probe" label="视频探测信息" />
              <AssetLink manifest={manifest} assetKey="video_frame_selection" label="视频关键帧选择" />
              <AssetLink manifest={manifest} assetKey="video_keyframe_timing" label="视频关键帧耗时" />
              <AssetLink manifest={manifest} assetKey="video_registration_diagnostics" label="视频注册诊断" />
              <AssetLink manifest={manifest} assetKey="video_initial_registration_expansion" label="视频初始注册扩展诊断" />
              <AssetLink manifest={manifest} assetKey="video_registration_recovery" label="视频注册空洞恢复诊断" />
              <AssetLink manifest={manifest} assetKey="colmap_timing" label="COLMAP 几何耗时" />
              <AssetLink manifest={manifest} assetKey="video_keyframe_contact_sheet" label="视频关键帧预览" />
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
            </details>
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
  if (mode === "video") {
    const file = files[0];
    if (file.size > 2 * 1024 ** 3) {
      return "视频文件不能超过 2 GiB。";
    }
    if (!/\.(mp4|mov|m4v|webm)$/i.test(file.name)) {
      return "视频必须使用 MP4、MOV、M4V 或 WebM 格式。";
    }
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
    unavailable: "不可用",
    passed: "通过",
    recovered: "已恢复",
    not_needed: "无需恢复"
  };
  return value ? (labels[value] ?? formatPolicy(value)) : "-";
}

function formatStage(value: string | undefined) {
  const labels: Record<string, string> = {
    queued: "等待执行",
    validating: "校验输入",
    video_probing: "探测视频与方向",
    video_frame_scoring: "分析候选视频帧",
    video_frame_extraction: "生成视频关键帧",
    vggt_ba_descriptors: "VGGT-BA 图像关系描述",
    vggt_ba_windows: "VGGT-BA 分批相机与局部 BA",
    vggt_ba_recovery: "VGGT-BA 弱帧连通恢复",
    vggt_ba_pose_graph: "VGGT-BA 全局窗口图",
    vggt_ba_feature_extraction: "VGGT 初始化后的 COLMAP 特征提取",
    vggt_ba_feature_matching: "VGGT 初始化后的 COLMAP 特征匹配",
    vggt_ba_global_triangulation: "VGGT 相机全局三角化",
    vggt_ba_image_registration: "COLMAP 补注册弱帧",
    vggt_ba_global_bundle_adjustment: "VGGT 相机全局 BA",
    colmap_fallback_mapping: "普通 COLMAP 几何回退",
    gaussian_vggt_postprocess: "VGGT Train-depth 高斯清理",
    gaussian_vggt_filtered_validation: "VGGT 清理后验证",
    gaussian_vggt_filtered_export: "VGGT 清理后导出",
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
    : trainer.id === "mcmc"
      ? "MCMC v1（实验，gsplat）"
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
    colmap: "COLMAP（默认）",
    incremental: "增量式 Mapper",
    global_recovery_v1: "Global Mapper 恢复",
    incremental_core_repair_v1: "增量式 healthy-core 恢复",
    vggt_ba: "VGGT + BA（实验）",
    none: "关闭",
    vggt_visibility_v1: "VGGT Train-depth 清理（实验）",
    closed_graph_verified: "已验证非局部图边",
    open_trajectory_unverified: "未验证开放轨迹",
    vggt_graph_unusable_after_recovery: "有界恢复后 VGGT 窗口图仍不可用",
    vggt_seed_geometry_insufficient: "VGGT 初值无法形成足够稀疏几何",
    vggt_registration_gate_failed: "VGGT 初值的最终注册质量门未通过",
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

function formatPercentMetric(value: number | undefined) {
  return value === undefined ? "-" : `${(value * 100).toFixed(1)}%`;
}

function formatInteger(value: number | undefined) {
  return value === undefined ? "-" : value.toLocaleString("zh-CN");
}

function formatRatio(left: number | undefined, right: number | undefined) {
  return left === undefined || right === undefined
    ? "-"
    : `${left.toLocaleString("zh-CN")} / ${right.toLocaleString("zh-CN")}`;
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
