import { useEffect, useMemo, useRef, useState } from "react";
import {
  assetUrl,
  filterSfmImages,
  pairNeighbors,
  parseFeatureIndex,
  parseFeatureShard,
  parsePairIndex,
  parsePairShard,
  rankSfmImages,
  sampleDeterministic,
  type PairIndexEntry,
  type RankedSfmImage,
  type RankingMode,
  type RegistrationFilter,
  type SfmDiagnostics,
  type SfmImage,
  type SfmInspectionTab,
  type SfmPairDetail,
  type SplitFilter,
  type Vec3
} from "./sfmDiagnostics";

type MatchFilter = "all" | "inliers" | "outliers";
type Point = [number, number];
type LineLimit = 50 | 150 | 300;

type Props = {
  jobId: string;
  diagnostics: SfmDiagnostics;
  queryCenter: Vec3;
  queryForward: Vec3;
  initialTab: SfmInspectionTab;
  onTabChange: (tab: SfmInspectionTab) => void;
  onClose: () => void;
};

export function SfmInspectionPanel({
  jobId,
  diagnostics,
  queryCenter,
  queryForward,
  initialTab,
  onTabChange,
  onClose
}: Props) {
  const cache = useRef(new Map<string, unknown>());
  const [rankingMode, setRankingMode] = useState<RankingMode>("view");
  const [tab, setTab] = useState<SfmInspectionTab>(initialTab);
  const [selectedRunId, setSelectedRunId] = useState(diagnostics.default_run_id);
  const [selectedImageId, setSelectedImageId] = useState(
    diagnostics.images.find((image) => image.registered)?.colmap_image_id ??
      diagnostics.images[0]?.colmap_image_id ??
      0
  );
  const [frameQuery, setFrameQuery] = useState("");
  const [registrationFilter, setRegistrationFilter] = useState<RegistrationFilter>("all");
  const [splitFilter, setSplitFilter] = useState<SplitFilter>("all");
  const [showKeypoints, setShowKeypoints] = useState(true);
  const [featurePoints, setFeaturePoints] = useState(new Map<number, Point[]>());
  const [pairIndex, setPairIndex] = useState<PairIndexEntry[] | null>(null);
  const [selectedPairKey, setSelectedPairKey] = useState<string | null>(null);
  const [pairDetail, setPairDetail] = useState<SfmPairDetail | null>(null);
  const [matchFilter, setMatchFilter] = useState<MatchFilter>("all");
  const [lineLimit, setLineLimit] = useState<LineLimit>(150);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  const run =
    diagnostics.runs.find((item) => item.run_id === selectedRunId) ??
    diagnostics.runs.find((item) => item.run_id === diagnostics.default_run_id)!;
  const ranked = useMemo(
    () => rankSfmImages(diagnostics, queryCenter, queryForward, rankingMode),
    [diagnostics, queryCenter, queryForward, rankingMode]
  );
  const filteredImages = useMemo(
    () =>
      filterSfmImages(
        diagnostics.images,
        frameQuery,
        registrationFilter,
        splitFilter
      ),
    [diagnostics.images, frameQuery, registrationFilter, splitFilter]
  );
  const selectedImage =
    diagnostics.images.find((image) => image.colmap_image_id === selectedImageId) ??
    filteredImages[0] ??
    diagnostics.images[0] ??
    null;
  const frameOptions = useMemo(() => {
    const first = filteredImages.slice(0, 200);
    if (!selectedImage || first.some((image) => image.colmap_image_id === selectedImage.colmap_image_id)) {
      return first;
    }
    return [selectedImage, ...first.slice(0, 199)];
  }, [filteredImages, selectedImage]);
  const neighbors = useMemo(
    () => (pairIndex && selectedImage ? pairNeighbors(pairIndex, selectedImage.colmap_image_id) : []),
    [pairIndex, selectedImage]
  );
  const activeNeighbor =
    neighbors.find((neighbor) => neighbor.entry.pair_key === selectedPairKey) ??
    neighbors[0] ??
    null;
  const pairEntry = activeNeighbor?.entry ?? null;
  const imageById = useMemo(
    () => new Map(diagnostics.images.map((image) => [image.colmap_image_id, image])),
    [diagnostics.images]
  );
  const pairImages = pairEntry
    ? ([imageById.get(pairEntry.image_ids[0]), imageById.get(pairEntry.image_ids[1])].filter(
        Boolean
      ) as SfmImage[])
    : [];
  const registeredCount = diagnostics.images.filter((image) => image.registered).length;

  useEffect(() => {
    setTab(initialTab);
  }, [initialTab]);

  useEffect(() => {
    if (filteredImages.length > 0 && !filteredImages.some((image) => image.colmap_image_id === selectedImageId)) {
      setSelectedImageId(filteredImages[0].colmap_image_id);
      setSelectedPairKey(null);
    }
  }, [filteredImages, selectedImageId]);

  useEffect(() => {
    if (ranked[0]) {
      setSelectedImageId(ranked[0].image.colmap_image_id);
      setSelectedPairKey(null);
    }
  }, [queryCenter, queryForward]);

  useEffect(() => {
    setFeaturePoints(new Map());
    setPairIndex(null);
    setSelectedPairKey(null);
    setPairDetail(null);
  }, [run.run_id]);

  useEffect(() => {
    if (tab !== "matches") {
      return;
    }
    const controller = new AbortController();
    let cancelled = false;
    setLoading(true);
    setMessage("");
    void loadJson(jobId, run.pair_index_path, cache.current, controller.signal)
      .then(parsePairIndex)
      .then((index) => {
        if (!cancelled) setPairIndex(index);
      })
      .catch((error) => {
        if (!cancelled) setMessage(error instanceof Error ? error.message : "图对索引加载失败");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [jobId, run.pair_index_path, tab]);

  useEffect(() => {
    setPairDetail(null);
    if (tab !== "matches" || !pairEntry || pairEntry.candidate_match_count === 0) {
      return;
    }
    const controller = new AbortController();
    let cancelled = false;
    setLoading(true);
    setMessage("");
    void loadJson(jobId, pairEntry.detail_shard, cache.current, controller.signal)
      .then((payload) => parsePairShard(payload, pairEntry.pair_key))
      .then((detail) => {
        if (!cancelled) setPairDetail(detail);
      })
      .catch((error) => {
        if (!cancelled) setMessage(error instanceof Error ? error.message : "匹配明细加载失败");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [jobId, pairEntry?.detail_shard, pairEntry?.pair_key, pairEntry?.candidate_match_count, tab]);

  useEffect(() => {
    const wanted = new Set<number>();
    if (tab === "keypoints" && selectedImage) {
      wanted.add(selectedImage.colmap_image_id);
    }
    if (tab === "matches") {
      pairImages.forEach((image) => wanted.add(image.colmap_image_id));
    }
    if (wanted.size === 0) {
      return;
    }
    const controller = new AbortController();
    let cancelled = false;
    setLoading(true);
    setMessage("");
    void loadFeaturePoints(jobId, run.feature_index_path, wanted, cache.current, controller.signal)
      .then((points) => {
        if (!cancelled) {
          setFeaturePoints((current) => new Map([...current, ...points]));
        }
      })
      .catch((error) => {
        if (!cancelled) setMessage(error instanceof Error ? error.message : "关键点加载失败");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [jobId, pairEntry?.pair_key, run.feature_index_path, selectedImage?.colmap_image_id, tab]);

  const selectTab = (next: SfmInspectionTab) => {
    setTab(next);
    onTabChange(next);
  };

  const openFrame = (image: SfmImage, next: SfmInspectionTab) => {
    setSelectedImageId(image.colmap_image_id);
    setFrameQuery("");
    setRegistrationFilter("all");
    setSplitFilter("all");
    setSelectedPairKey(null);
    selectTab(next);
  };

  return (
    <aside className="sfm-inspector" aria-label="SfM 输入视图诊断">
      <header className="sfm-inspector-heading">
        <div>
          <strong>重建证据 · 输入与匹配</strong>
          <span>
            {registeredCount.toLocaleString()} / {diagnostics.images.length.toLocaleString()} 张已注册 · 任意单位（非米制）
          </span>
        </div>
        <div className="sfm-inspector-heading-actions">
          {diagnostics.runs.length > 1 && (
            <label>
              <span>算法运行</span>
              <select value={run.run_id} onChange={(event) => setSelectedRunId(event.target.value)}>
                {diagnostics.runs.map((item) => (
                  <option key={item.run_id} value={item.run_id}>
                    {item.detector.name} + {item.matcher.name}
                  </option>
                ))}
              </select>
            </label>
          )}
          <button className="viewer-tool-button" onClick={onClose} type="button">关闭</button>
        </div>
      </header>

      <div className="sfm-run-provenance">
        <span>Detector: {run.detector.implementation}/{run.detector.name} {run.detector.version}</span>
        <span>Matcher: {run.matcher.implementation}/{run.matcher.name} {run.matcher.version}</span>
      </div>

      <div className="sfm-inspector-controls">
        <div className="variant-toggle" role="tablist" aria-label="SfM 诊断图层">
          {(["nearest", "keypoints", "matches"] as SfmInspectionTab[]).map((value) => (
            <button
              aria-selected={tab === value}
              className={tab === value ? "active" : ""}
              key={value}
              onClick={() => selectTab(value)}
              role="tab"
              type="button"
            >
              {value === "nearest" ? "当前视角" : value === "keypoints" ? "关键点" : "SfM 匹配"}
            </button>
          ))}
        </div>
        {tab === "nearest" && (
          <div className="variant-toggle" role="group" aria-label="输入视图排序方式">
            <button className={rankingMode === "view" ? "active" : ""} onClick={() => setRankingMode("view")} type="button">视角最相似</button>
            <button className={rankingMode === "position" ? "active" : ""} onClick={() => setRankingMode("position")} type="button">位置最近</button>
          </div>
        )}
      </div>

      {tab === "nearest" && ranked.some((entry) => entry.confidence === "poor") && (
        <div className="sfm-inspector-warning" role="status">
          最近结果包含距离过远或朝向差过大的图片；排名最近不代表已覆盖当前视角。
        </div>
      )}

      {tab === "nearest" && (
        <div className="sfm-image-grid">
          {ranked.map((entry, index) => (
            <article className="sfm-image-card" key={entry.image.frame_uid}>
              <div className="sfm-image-label">{["A", "B", "C"][index]}</div>
              <KeypointImage
                alt={`输入视图 ${["A", "B", "C"][index]}：${entry.image.name}`}
                image={entry.image}
                imageUrl={assetUrl(jobId, entry.image.path)}
                points={null}
              />
              <ImageMetrics diagnostics={diagnostics} entry={entry} />
              <div className="sfm-card-actions">
                <button type="button" onClick={() => openFrame(entry.image, "keypoints")}>查看关键点</button>
                <button type="button" onClick={() => openFrame(entry.image, "matches")}>查看匹配</button>
              </div>
            </article>
          ))}
        </div>
      )}

      {tab === "keypoints" && (
        <section className="sfm-frame-workspace">
          <FramePicker
            filteredImages={filteredImages}
            frameOptions={frameOptions}
            query={frameQuery}
            registration={registrationFilter}
            selectedImageId={selectedImage?.colmap_image_id ?? 0}
            split={splitFilter}
            totalImages={diagnostics.images.length}
            onQueryChange={setFrameQuery}
            onRegistrationChange={setRegistrationFilter}
            onSelectImage={(id) => {
              setSelectedImageId(id);
              setSelectedPairKey(null);
            }}
            onSplitChange={setSplitFilter}
          />
          {selectedImage ? (
            <div className="sfm-frame-detail">
              <div className="sfm-frame-detail-heading">
                <div>
                  <strong>{selectedImage.name}</strong>
                  <span>COLMAP image #{selectedImage.colmap_image_id}</span>
                </div>
                <button
                  className={showKeypoints ? "viewer-tool-button active" : "viewer-tool-button"}
                  onClick={() => setShowKeypoints((current) => !current)}
                  type="button"
                >
                  {showKeypoints ? "隐藏关键点" : "显示关键点"}
                </button>
              </div>
              <KeypointImage
                alt={`输入帧：${selectedImage.name}`}
                image={selectedImage}
                imageUrl={assetUrl(jobId, selectedImage.path)}
                points={showKeypoints ? featurePoints.get(selectedImage.colmap_image_id) ?? [] : null}
              />
              <FrameMetrics image={selectedImage} />
              <button className="viewer-tool-button sfm-next-action" onClick={() => selectTab("matches")} type="button">
                查看该帧的已测试图对
              </button>
            </div>
          ) : (
            <div className="sfm-empty">没有符合筛选条件的输入帧。</div>
          )}
        </section>
      )}

      {tab === "matches" && (
        <section className="sfm-match-section">
          <FramePicker
            filteredImages={filteredImages}
            frameOptions={frameOptions}
            query={frameQuery}
            registration={registrationFilter}
            selectedImageId={selectedImage?.colmap_image_id ?? 0}
            split={splitFilter}
            totalImages={diagnostics.images.length}
            onQueryChange={setFrameQuery}
            onRegistrationChange={setRegistrationFilter}
            onSelectImage={(id) => {
              setSelectedImageId(id);
              setSelectedPairKey(null);
            }}
            onSplitChange={setSplitFilter}
          />
          {pairIndex === null ? (
            <div className="sfm-empty">正在加载 matcher 图对索引…</div>
          ) : !selectedImage ? (
            <div className="sfm-empty">请选择一张主帧。</div>
          ) : neighbors.length === 0 ? (
            <div className="sfm-empty">该帧没有 matcher 测试记录；这不等于测试后零匹配。</div>
          ) : (
            <>
              <div className="sfm-match-controls">
                <label className="sfm-pair-select">
                  <span>已测试邻接图对（{neighbors.length.toLocaleString()}）</span>
                  <select
                    value={activeNeighbor?.entry.pair_key ?? ""}
                    onChange={(event) => setSelectedPairKey(event.target.value)}
                  >
                    {neighbors.map((neighbor) => {
                      const image = imageById.get(neighbor.neighbor_image_id);
                      return (
                        <option key={neighbor.entry.pair_key} value={neighbor.entry.pair_key}>
                          {image?.name ?? `image #${neighbor.neighbor_image_id}`} · 内点 {neighbor.entry.inlier_count} / {neighbor.entry.candidate_match_count} ({formatPercent(neighbor.inlier_rate)})
                        </option>
                      );
                    })}
                  </select>
                </label>
                <div className="sfm-match-filter-row">
                  <div className="variant-toggle" role="group" aria-label="匹配类型">
                    {(["all", "inliers", "outliers"] as MatchFilter[]).map((value) => (
                      <button className={matchFilter === value ? "active" : ""} key={value} onClick={() => setMatchFilter(value)} type="button">
                        {value === "all" ? "全部" : value === "inliers" ? "内点" : "外点"}
                      </button>
                    ))}
                  </div>
                  <label>
                    <span>显示线数</span>
                    <select value={lineLimit} onChange={(event) => setLineLimit(Number(event.target.value) as LineLimit)}>
                      <option value={50}>50</option>
                      <option value={150}>150</option>
                      <option value={300}>300</option>
                    </select>
                  </label>
                </div>
              </div>
              {pairEntry?.candidate_match_count === 0 ? (
                <div className="sfm-empty">该图像对已测试，但没有候选匹配。</div>
              ) : pairEntry && pairDetail && pairImages.length === 2 ? (
                <>
                  <div className="sfm-match-summary">
                    候选 {pairEntry.candidate_match_count.toLocaleString()} · 内点 {pairEntry.inlier_count.toLocaleString()} · 外点 {(pairEntry.candidate_match_count - pairEntry.inlier_count).toLocaleString()} · 内点率 {formatPercent(pairEntry.inlier_count / pairEntry.candidate_match_count)} · geometric config {pairEntry.geometric_config}
                  </div>
                  <MatchCanvas
                    detail={pairDetail}
                    filter={matchFilter}
                    left={pairImages[0]}
                    leftPoints={featurePoints.get(pairImages[0].colmap_image_id) ?? []}
                    leftUrl={assetUrl(jobId, pairImages[0].path)}
                    lineLimit={lineLimit}
                    right={pairImages[1]}
                    rightPoints={featurePoints.get(pairImages[1].colmap_image_id) ?? []}
                    rightUrl={assetUrl(jobId, pairImages[1].path)}
                  />
                  <div className="sfm-match-legend" aria-label="匹配图例">
                    <span><i className="match-legend-line inlier" />几何内点（实线）</span>
                    <span><i className="match-legend-line outlier" />候选外点（虚线）</span>
                  </div>
                </>
              ) : (
                <div className="sfm-empty">正在加载匹配明细与关键点…</div>
              )}
            </>
          )}
        </section>
      )}

      {(loading || message) && <div className="sfm-inspector-status" role="status">{message || "正在加载结构化诊断数据…"}</div>}
    </aside>
  );
}

function FramePicker({
  filteredImages,
  frameOptions,
  query,
  registration,
  selectedImageId,
  split,
  totalImages,
  onQueryChange,
  onRegistrationChange,
  onSelectImage,
  onSplitChange
}: {
  filteredImages: SfmImage[];
  frameOptions: SfmImage[];
  query: string;
  registration: RegistrationFilter;
  selectedImageId: number;
  split: SplitFilter;
  totalImages: number;
  onQueryChange: (value: string) => void;
  onRegistrationChange: (value: RegistrationFilter) => void;
  onSelectImage: (id: number) => void;
  onSplitChange: (value: SplitFilter) => void;
}) {
  return (
    <div className="sfm-frame-picker">
      <label className="sfm-frame-search">
        <span>搜索帧</span>
        <input
          type="search"
          placeholder="文件名、COLMAP ID 或秒数"
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
        />
      </label>
      <label>
        <span>注册状态</span>
        <select value={registration} onChange={(event) => onRegistrationChange(event.target.value as RegistrationFilter)}>
          <option value="all">全部</option>
          <option value="registered">已注册</option>
          <option value="unregistered">未注册</option>
        </select>
      </label>
      <label>
        <span>训练切分</span>
        <select value={split} onChange={(event) => onSplitChange(event.target.value as SplitFilter)}>
          <option value="all">全部</option>
          <option value="train">Train</option>
          <option value="validation">Validation</option>
          <option value="test">Test</option>
          <option value="unassigned">未分配</option>
        </select>
      </label>
      <label className="sfm-frame-select">
        <span>输入帧 · {filteredImages.length.toLocaleString()} / {totalImages.toLocaleString()}</span>
        <select
          disabled={frameOptions.length === 0}
          value={frameOptions.length === 0 ? "" : selectedImageId}
          onChange={(event) => onSelectImage(Number(event.target.value))}
        >
          {frameOptions.length === 0 && <option value="">没有符合条件的帧</option>}
          {frameOptions.map((image) => (
            <option key={image.colmap_image_id} value={image.colmap_image_id}>
              {image.registered ? "✓" : "×"} {formatTime(image.source_time_seconds)} · {image.name}
            </option>
          ))}
        </select>
      </label>
      {filteredImages.length > 200 && (
        <small>列表显示前 200 项；继续输入文件名、ID 或时间可精确定位全部帧。</small>
      )}
    </div>
  );
}

function KeypointImage({
  alt,
  image,
  imageUrl,
  points
}: {
  alt: string;
  image: SfmImage;
  imageUrl: string;
  points: Point[] | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || points === null) return;
    canvas.width = image.width;
    canvas.height = image.height;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#2a78d6";
    const radius = Math.max(1.6, canvas.width / 500);
    for (const [x, y] of points) {
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.fill();
    }
  }, [image.height, image.width, points]);
  return (
    <div className="sfm-keypoint-image">
      <img alt={alt} src={imageUrl} />
      {points !== null && <canvas aria-label={`${points.length} 个 SIFT 关键点`} ref={canvasRef} />}
    </div>
  );
}

function ImageMetrics({ diagnostics, entry }: { diagnostics: SfmDiagnostics; entry: RankedSfmImage }) {
  const time = entry.image.source_time_seconds;
  const nearbyFailures =
    time === null
      ? 0
      : diagnostics.images.filter(
          (image) => !image.registered && image.source_time_seconds !== null && Math.abs(image.source_time_seconds - time) <= 2
        ).length;
  return (
    <dl className="sfm-image-metrics">
      <div><dt>距离</dt><dd>{entry.distance.toFixed(3)}</dd></div>
      <div><dt>朝向差</dt><dd>{entry.angleDegrees.toFixed(1)}°</dd></div>
      <div><dt>视锥关系</dt><dd>{formatFrustum(entry.frustumRelation)}</dd></div>
      <div><dt>可信度</dt><dd>{formatConfidence(entry.confidence)}</dd></div>
      <div><dt>视频时间</dt><dd>{formatTime(time)}</dd></div>
      <div><dt>训练切分</dt><dd>{entry.image.split ?? "-"}</dd></div>
      <div><dt>关键点</dt><dd>{entry.image.feature_count.toLocaleString()}</dd></div>
      <div><dt>±2s 未注册</dt><dd>{nearbyFailures}</dd></div>
    </dl>
  );
}

function FrameMetrics({ image }: { image: SfmImage }) {
  return (
    <dl className="sfm-frame-metrics">
      <div><dt>注册</dt><dd>{image.registered ? "已注册" : "未注册（无最终位姿）"}</dd></div>
      <div><dt>视频时间</dt><dd>{formatTime(image.source_time_seconds)}</dd></div>
      <div><dt>训练切分</dt><dd>{image.split ?? "未分配"}</dd></div>
      <div><dt>关键点</dt><dd>{image.feature_count.toLocaleString()}</dd></div>
      <div><dt>分辨率</dt><dd>{image.width} × {image.height}</dd></div>
      <div><dt>帧 UID</dt><dd title={image.frame_uid}>{image.frame_uid.slice(0, 12)}…</dd></div>
    </dl>
  );
}

function MatchCanvas({
  detail,
  filter,
  left,
  leftPoints,
  leftUrl,
  lineLimit,
  right,
  rightPoints,
  rightUrl
}: {
  detail: SfmPairDetail;
  filter: MatchFilter;
  left: SfmImage;
  leftPoints: Point[];
  leftUrl: string;
  lineLimit: LineLimit;
  right: SfmImage;
  rightPoints: Point[];
  rightUrl: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hovered, setHovered] = useState("");
  const [drawError, setDrawError] = useState("");
  const segmentsRef = useRef<Array<{ start: Point; end: Point; label: string }>>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let cancelled = false;
    setDrawError("");
    void Promise.all([loadImage(leftUrl), loadImage(rightUrl)])
      .then(([leftImage, rightImage]) => {
        if (cancelled) return;
        const paneWidth = 570;
        const gap = 60;
        const leftScale = paneWidth / left.width;
        const rightScale = paneWidth / right.width;
        const leftHeight = left.height * leftScale;
        const rightHeight = right.height * rightScale;
        canvas.width = paneWidth * 2 + gap;
        canvas.height = Math.ceil(Math.max(leftHeight, rightHeight));
        const context = canvas.getContext("2d");
        if (!context) return;
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.drawImage(leftImage, 0, 0, paneWidth, leftHeight);
        context.drawImage(rightImage, paneWidth + gap, 0, paneWidth, rightHeight);
        context.font = "bold 16px ui-monospace, monospace";
        context.fillStyle = "#ffffff";
        context.fillText(`#${left.colmap_image_id}`, 12, 24);
        context.fillText(`#${right.colmap_image_id}`, paneWidth + gap + 12, 24);
        const sampled = sampleDeterministic(selectMatches(detail, filter), lineLimit);
        const segments: Array<{ start: Point; end: Point; label: string }> = [];
        for (const match of sampled) {
          const leftPoint = leftPoints[match.indices[0]];
          const rightPoint = rightPoints[match.indices[1]];
          if (!leftPoint || !rightPoint) continue;
          const start: Point = [leftPoint[0] * leftScale, leftPoint[1] * leftScale];
          const end: Point = [paneWidth + gap + rightPoint[0] * rightScale, rightPoint[1] * rightScale];
          const color = match.kind === "inlier" ? "#2a78d6" : "#eb6834";
          context.beginPath();
          context.moveTo(...start);
          context.lineTo(...end);
          context.strokeStyle = color;
          context.lineWidth = 1.4;
          context.setLineDash(match.kind === "inlier" ? [] : [7, 5]);
          context.globalAlpha = 0.68;
          context.stroke();
          context.setLineDash([]);
          context.fillStyle = color;
          context.beginPath();
          context.arc(start[0], start[1], 2.3, 0, Math.PI * 2);
          context.arc(end[0], end[1], 2.3, 0, Math.PI * 2);
          context.fill();
          segments.push({
            start,
            end,
            label: `${match.kind === "inlier" ? "几何内点" : "候选外点"} · ${match.indices[0]} ↔ ${match.indices[1]}`
          });
        }
        context.globalAlpha = 1;
        context.setLineDash([]);
        segmentsRef.current = segments;
      })
      .catch(() => {
        segmentsRef.current = [];
        if (!cancelled) setDrawError("输入图片加载失败，无法绘制匹配。");
      });
    return () => {
      cancelled = true;
      segmentsRef.current = [];
    };
  }, [detail, filter, left, leftPoints, leftUrl, lineLimit, right, rightPoints, rightUrl]);

  function onPointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = event.currentTarget;
    const bounds = canvas.getBoundingClientRect();
    const point: Point = [
      ((event.clientX - bounds.left) / bounds.width) * canvas.width,
      ((event.clientY - bounds.top) / bounds.height) * canvas.height
    ];
    const threshold = (8 / Math.max(bounds.width, 1)) * canvas.width;
    const match = segmentsRef.current.find((segment) => pointSegmentDistance(point, segment.start, segment.end) <= threshold);
    setHovered(match?.label ?? "");
  }

  return (
    <div className="sfm-match-canvas-wrap">
      <canvas
        aria-label={`${left.name} 与 ${right.name} 的 SfM 匹配图`}
        onPointerLeave={() => setHovered("")}
        onPointerMove={onPointerMove}
        ref={canvasRef}
      />
      {hovered && <span className="sfm-match-tooltip">{hovered}</span>}
      {drawError && <span className="sfm-match-canvas-error" role="status">{drawError}</span>}
    </div>
  );
}

async function loadFeaturePoints(
  jobId: string,
  indexPath: string,
  wanted: Set<number>,
  cache: Map<string, unknown>,
  signal: AbortSignal
): Promise<Map<number, Point[]>> {
  const index = parseFeatureIndex(await loadJson(jobId, indexPath, cache, signal));
  const entries = index.filter((entry) => wanted.has(entry.image_id));
  const shards = new Map<string, unknown>();
  await Promise.all(
    [...new Set(entries.map((entry) => entry.detail_shard))].map(async (path) => {
      shards.set(path, await loadJson(jobId, path, cache, signal));
    })
  );
  return new Map(
    entries.map((entry) => [
      entry.image_id,
      parseFeatureShard(shards.get(entry.detail_shard), entry.image_id)
    ])
  );
}

async function loadJson(
  jobId: string,
  path: string,
  cache: Map<string, unknown>,
  signal: AbortSignal
): Promise<unknown> {
  const url = assetUrl(jobId, path);
  if (cache.has(url)) return cache.get(url);
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`诊断资产加载失败：${response.status}`);
  const value: unknown = await response.json();
  cache.set(url, value);
  return value;
}

function selectMatches(detail: SfmPairDetail, filter: MatchFilter) {
  const inliers = detail.inliers.map((indices) => ({ kind: "inlier" as const, indices }));
  const outliers = detail.outliers.map((indices) => ({ kind: "outlier" as const, indices }));
  return filter === "inliers" ? inliers : filter === "outliers" ? outliers : [...inliers, ...outliers];
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("输入图片加载失败"));
    image.src = url;
  });
}

function pointSegmentDistance(point: Point, start: Point, end: Point): number {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) return Math.hypot(point[0] - start[0], point[1] - start[1]);
  const t = Math.max(0, Math.min(1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / lengthSquared));
  return Math.hypot(point[0] - (start[0] + t * dx), point[1] - (start[1] + t * dy));
}

function formatFrustum(value: RankedSfmImage["frustumRelation"]) {
  return value === "aligned" ? "朝向中心" : value === "edge" ? "朝向边缘" : "朝向外部";
}

function formatConfidence(value: RankedSfmImage["confidence"]) {
  return value === "good" ? "良好" : value === "limited" ? "有限" : "较差";
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatTime(value: number | null) {
  return value === null ? "-" : `${value.toFixed(2)} s`;
}
