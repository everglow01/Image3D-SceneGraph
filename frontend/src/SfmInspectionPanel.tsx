import { useEffect, useMemo, useRef, useState } from "react";
import {
  assetUrl,
  pairKey,
  parseFeatureIndex,
  parseFeatureShard,
  parsePairIndex,
  parsePairShard,
  rankSfmImages,
  type PairIndexEntry,
  type RankedSfmImage,
  type RankingMode,
  type SfmDiagnostics,
  type SfmPairDetail,
  type Vec3
} from "./sfmDiagnostics";

type InspectionTab = "original" | "keypoints" | "matches";
type MatchFilter = "all" | "inliers" | "outliers";
type Point = [number, number];

type Props = {
  jobId: string;
  diagnostics: SfmDiagnostics;
  queryCenter: Vec3;
  queryForward: Vec3;
  onClose: () => void;
};

type PairOption = {
  label: string;
  leftLabel: string;
  rightLabel: string;
  left: RankedSfmImage;
  right: RankedSfmImage;
  key: string;
};

export function SfmInspectionPanel({
  jobId,
  diagnostics,
  queryCenter,
  queryForward,
  onClose
}: Props) {
  const cache = useRef(new Map<string, unknown>());
  const [rankingMode, setRankingMode] = useState<RankingMode>("view");
  const [tab, setTab] = useState<InspectionTab>("original");
  const [featurePoints, setFeaturePoints] = useState(new Map<number, Point[]>());
  const [pairIndex, setPairIndex] = useState<PairIndexEntry[] | null>(null);
  const [selectedPair, setSelectedPair] = useState(0);
  const [pairDetail, setPairDetail] = useState<SfmPairDetail | null>(null);
  const [matchFilter, setMatchFilter] = useState<MatchFilter>("all");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const ranked = useMemo(
    () => rankSfmImages(diagnostics, queryCenter, queryForward, rankingMode),
    [diagnostics, queryCenter, queryForward, rankingMode]
  );
  const labels = ["A", "B", "C"];
  const pairs = useMemo(() => pairOptions(ranked, labels), [ranked]);
  const activePair = pairs[selectedPair] ?? pairs[0] ?? null;
  const run = diagnostics.runs.find((item) => item.run_id === diagnostics.default_run_id)!;

  useEffect(() => {
    setSelectedPair(0);
    setPairDetail(null);
  }, [rankingMode, queryCenter, queryForward]);

  useEffect(() => {
    if (tab === "original" || ranked.length === 0) {
      return;
    }
    const controller = new AbortController();
    let cancelled = false;
    setLoading(true);
    setMessage("");
    void loadFeaturePoints(jobId, run.feature_index_path, ranked, cache.current, controller.signal)
      .then((points) => {
        if (!cancelled) {
          setFeaturePoints(points);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setMessage(error instanceof Error ? error.message : "关键点加载失败");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [jobId, ranked, run.feature_index_path, tab]);

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
        if (!cancelled) {
          setPairIndex(index);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setMessage(error instanceof Error ? error.message : "图对索引加载失败");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [jobId, run.pair_index_path, tab]);

  const pairEntry = activePair
    ? pairIndex?.find((entry) => entry.pair_key === activePair.key) ?? null
    : null;

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
        if (!cancelled) {
          setPairDetail(detail);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setMessage(error instanceof Error ? error.message : "匹配明细加载失败");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [jobId, pairEntry?.detail_shard, pairEntry?.pair_key, pairEntry?.candidate_match_count, tab]);

  const registeredCount = diagnostics.images.filter((image) => image.registered).length;

  return (
    <aside className="sfm-inspector" aria-label="SfM 输入视图诊断">
      <header className="sfm-inspector-heading">
        <div>
          <strong>输入视图诊断</strong>
          <span>{registeredCount} 张已注册图片 · 坐标为归一化任意单位</span>
        </div>
        <button className="viewer-tool-button" onClick={onClose} type="button">关闭</button>
      </header>

      <div className="sfm-inspector-controls">
        <div className="variant-toggle" role="group" aria-label="输入视图排序方式">
          <button className={rankingMode === "view" ? "active" : ""} onClick={() => setRankingMode("view")} type="button">视角最相似</button>
          <button className={rankingMode === "position" ? "active" : ""} onClick={() => setRankingMode("position")} type="button">位置最近</button>
        </div>
        <div className="variant-toggle" role="tablist" aria-label="SfM 诊断图层">
          {(["original", "keypoints", "matches"] as InspectionTab[]).map((value) => (
            <button
              aria-selected={tab === value}
              className={tab === value ? "active" : ""}
              key={value}
              onClick={() => setTab(value)}
              role="tab"
              type="button"
            >
              {value === "original" ? "原始帧" : value === "keypoints" ? "关键点" : "SfM 匹配"}
            </button>
          ))}
        </div>
      </div>

      {ranked.some((entry) => entry.confidence === "poor") && (
        <div className="sfm-inspector-warning" role="status">
          ⚠ 最近结果中存在距离过远或朝向差过大的图片；“排名最近”不代表可靠覆盖。
        </div>
      )}

      {tab !== "matches" ? (
        <div className="sfm-image-grid">
          {ranked.map((entry, index) => (
            <article className="sfm-image-card" key={entry.image.frame_uid}>
              <div className="sfm-image-label">{labels[index]}</div>
              <KeypointImage
                alt={`输入视图 ${labels[index]}：${entry.image.name}`}
                image={entry}
                imageUrl={assetUrl(jobId, entry.image.path)}
                points={tab === "keypoints" ? featurePoints.get(entry.image.colmap_image_id) ?? [] : null}
              />
              <ImageMetrics diagnostics={diagnostics} entry={entry} />
            </article>
          ))}
        </div>
      ) : (
        <section className="sfm-match-section">
          <div className="sfm-match-controls">
            <div className="variant-toggle" role="group" aria-label="匹配图像对">
              {pairs.map((pair, index) => (
                <button className={selectedPair === index ? "active" : ""} key={pair.label} onClick={() => setSelectedPair(index)} type="button">{pair.label}</button>
              ))}
            </div>
            <div className="variant-toggle" role="group" aria-label="匹配类型">
              {(["all", "inliers", "outliers"] as MatchFilter[]).map((value) => (
                <button className={matchFilter === value ? "active" : ""} key={value} onClick={() => setMatchFilter(value)} type="button">
                  {value === "all" ? "全部" : value === "inliers" ? "内点" : "外点"}
                </button>
              ))}
            </div>
          </div>
          {!activePair ? (
            <div className="sfm-empty">注册图片不足两张，无法查看图对。</div>
          ) : pairIndex === null ? (
            <div className="sfm-empty">正在加载图对索引…</div>
          ) : pairEntry === null ? (
            <div className="sfm-empty">该图像对未被 matcher 测试；不等于测试后零匹配。</div>
          ) : pairEntry.candidate_match_count === 0 ? (
            <div className="sfm-empty">该图像对已测试，但没有候选匹配。</div>
          ) : pairDetail ? (
            <>
              <div className="sfm-match-summary">
                候选 {pairEntry.candidate_match_count.toLocaleString()} · 内点 {pairEntry.inlier_count.toLocaleString()} · 内点率 {formatPercent(pairEntry.inlier_count / pairEntry.candidate_match_count)} · 画布最多显示 300 条
              </div>
              <MatchCanvas
                detail={pairDetail}
                filter={matchFilter}
                left={activePair.left}
                leftLabel={activePair.leftLabel}
                leftPoints={featurePoints.get(activePair.left.image.colmap_image_id) ?? []}
                leftUrl={assetUrl(jobId, activePair.left.image.path)}
                right={activePair.right}
                rightLabel={activePair.rightLabel}
                rightPoints={featurePoints.get(activePair.right.image.colmap_image_id) ?? []}
                rightUrl={assetUrl(jobId, activePair.right.image.path)}
              />
              <div className="sfm-match-legend" aria-label="匹配图例">
                <span><i className="match-legend-line inlier" />几何内点（实线）</span>
                <span><i className="match-legend-line outlier" />候选外点（虚线）</span>
              </div>
            </>
          ) : (
            <div className="sfm-empty">正在加载匹配明细…</div>
          )}
        </section>
      )}

      {(loading || message) && <div className="sfm-inspector-status" role="status">{message || "正在加载结构化诊断数据…"}</div>}
    </aside>
  );
}

function KeypointImage({
  alt,
  image,
  imageUrl,
  points
}: {
  alt: string;
  image: RankedSfmImage;
  imageUrl: string;
  points: Point[] | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || points === null) {
      return;
    }
    canvas.width = image.image.width;
    canvas.height = image.image.height;
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#2a78d6";
    const radius = Math.max(1.6, canvas.width / 500);
    for (const [x, y] of points) {
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.fill();
    }
  }, [image.image.height, image.image.width, points]);
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
      <div><dt>视频时间</dt><dd>{time === null ? "-" : `${time.toFixed(2)} s`}</dd></div>
      <div><dt>训练切分</dt><dd>{entry.image.split ?? "-"}</dd></div>
      <div><dt>关键点</dt><dd>{entry.image.feature_count.toLocaleString()}</dd></div>
      <div><dt>±2s 未注册</dt><dd>{nearbyFailures}</dd></div>
    </dl>
  );
}

function MatchCanvas({
  detail,
  filter,
  left,
  leftLabel,
  leftPoints,
  leftUrl,
  right,
  rightLabel,
  rightPoints,
  rightUrl
}: {
  detail: SfmPairDetail;
  filter: MatchFilter;
  left: RankedSfmImage;
  leftLabel: string;
  leftPoints: Point[];
  leftUrl: string;
  right: RankedSfmImage;
  rightLabel: string;
  rightPoints: Point[];
  rightUrl: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hovered, setHovered] = useState<string>("");
  const segmentsRef = useRef<Array<{ start: Point; end: Point; label: string }>>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    let cancelled = false;
    Promise.all([loadImage(leftUrl), loadImage(rightUrl)]).then(([leftImage, rightImage]) => {
      if (cancelled) {
        return;
      }
      const paneWidth = 570;
      const gap = 60;
      const leftScale = paneWidth / left.image.width;
      const rightScale = paneWidth / right.image.width;
      const leftHeight = left.image.height * leftScale;
      const rightHeight = right.image.height * rightScale;
      canvas.width = paneWidth * 2 + gap;
      canvas.height = Math.ceil(Math.max(leftHeight, rightHeight));
      const context = canvas.getContext("2d");
      if (!context) {
        return;
      }
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(leftImage, 0, 0, paneWidth, leftHeight);
      context.drawImage(rightImage, paneWidth + gap, 0, paneWidth, rightHeight);
      context.font = "bold 18px system-ui";
      context.fillStyle = "#ffffff";
      context.fillText(leftLabel, 12, 26);
      context.fillText(rightLabel, paneWidth + gap + 12, 26);
      const matches = selectMatches(detail, filter);
      const sampled = sample(matches, 300);
      const segments: Array<{ start: Point; end: Point; label: string }> = [];
      for (const match of sampled) {
        const leftPoint = leftPoints[match.indices[0]];
        const rightPoint = rightPoints[match.indices[1]];
        if (!leftPoint || !rightPoint) {
          continue;
        }
        const start: Point = [leftPoint[0] * leftScale, leftPoint[1] * leftScale];
        const end: Point = [paneWidth + gap + rightPoint[0] * rightScale, rightPoint[1] * rightScale];
        context.beginPath();
        context.moveTo(...start);
        context.lineTo(...end);
        context.strokeStyle = match.kind === "inlier" ? "#2a78d6" : "#eb6834";
        context.lineWidth = 1.4;
        context.setLineDash(match.kind === "inlier" ? [] : [7, 5]);
        context.globalAlpha = 0.68;
        context.stroke();
        segments.push({
          start,
          end,
          label: `${match.kind === "inlier" ? "几何内点" : "候选外点"} · ${match.indices[0]} ↔ ${match.indices[1]}`
        });
      }
      context.globalAlpha = 1;
      context.setLineDash([]);
      segmentsRef.current = segments;
    });
    return () => {
      cancelled = true;
      segmentsRef.current = [];
    };
  }, [detail, filter, left.image.height, left.image.width, leftPoints, leftUrl, right.image.height, right.image.width, rightPoints, rightUrl, leftLabel, rightLabel]);

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
        aria-label={`${leftLabel} 与 ${rightLabel} 的 SfM 匹配图`}
        onPointerLeave={() => setHovered("")}
        onPointerMove={onPointerMove}
        ref={canvasRef}
      />
      {hovered && <span className="sfm-match-tooltip">{hovered}</span>}
    </div>
  );
}

async function loadFeaturePoints(
  jobId: string,
  indexPath: string,
  ranked: RankedSfmImage[],
  cache: Map<string, unknown>,
  signal: AbortSignal
): Promise<Map<number, Point[]>> {
  const index = parseFeatureIndex(await loadJson(jobId, indexPath, cache, signal));
  const wanted = new Set(ranked.map((entry) => entry.image.colmap_image_id));
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
  if (cache.has(url)) {
    return cache.get(url);
  }
  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw new Error(`诊断资产加载失败：${response.status}`);
  }
  const value: unknown = await response.json();
  cache.set(url, value);
  return value;
}

function pairOptions(ranked: RankedSfmImage[], labels: string[]): PairOption[] {
  const result: PairOption[] = [];
  for (let left = 0; left < ranked.length; left += 1) {
    for (let right = left + 1; right < ranked.length; right += 1) {
      const first = ranked[left].image.colmap_image_id < ranked[right].image.colmap_image_id ? ranked[left] : ranked[right];
      const second = first === ranked[left] ? ranked[right] : ranked[left];
      result.push({
        label: `${labels[left]} ↔ ${labels[right]}`,
        leftLabel: first === ranked[left] ? labels[left] : labels[right],
        rightLabel: second === ranked[right] ? labels[right] : labels[left],
        left: first,
        right: second,
        key: pairKey(first.image.colmap_image_id, second.image.colmap_image_id)
      });
    }
  }
  return result;
}

function selectMatches(detail: SfmPairDetail, filter: MatchFilter) {
  const inliers = detail.inliers.map((indices) => ({ kind: "inlier" as const, indices }));
  const outliers = detail.outliers.map((indices) => ({ kind: "outlier" as const, indices }));
  return filter === "inliers" ? inliers : filter === "outliers" ? outliers : [...inliers, ...outliers];
}

function sample<T>(items: T[], maximum: number): T[] {
  if (items.length <= maximum) {
    return items;
  }
  return Array.from({ length: maximum }, (_, index) => items[Math.floor((index * items.length) / maximum)]);
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
  if (lengthSquared === 0) {
    return Math.hypot(point[0] - start[0], point[1] - start[1]);
  }
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
