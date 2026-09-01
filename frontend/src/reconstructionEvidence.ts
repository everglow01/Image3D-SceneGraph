export type EvidenceStageId = "input" | "matching" | "sparse" | "gaussian";

export type EvidenceStage = {
  id: EvidenceStageId;
  label: string;
  value: string;
  available: boolean;
  unavailableReason: string;
};

export type EvidenceSource = {
  hasDiagnostics: boolean;
  hasSparseGeometry: boolean;
  hasGaussian: boolean;
  imageCount?: number;
  registeredImageCount?: number;
  pairCount?: number;
  sparsePointCount?: number;
  gaussianCount?: number;
};

export function buildEvidenceStages(source: EvidenceSource): EvidenceStage[] {
  const diagnosticsReason = "此 job 未保存 SfM 前端诊断；需由新 job 在最终几何阶段导出。";
  return [
    {
      id: "input",
      label: "输入视图",
      value:
        source.imageCount === undefined
          ? "诊断不可用"
          : `${formatCount(source.registeredImageCount)} / ${formatCount(source.imageCount)} 注册`,
      available: source.hasDiagnostics,
      unavailableReason: diagnosticsReason
    },
    {
      id: "matching",
      label: "特征匹配",
      value: source.pairCount === undefined ? "诊断不可用" : `${formatCount(source.pairCount)} 图对`,
      available: source.hasDiagnostics,
      unavailableReason: diagnosticsReason
    },
    {
      id: "sparse",
      label: "稀疏几何",
      value:
        source.sparsePointCount === undefined
          ? source.hasSparseGeometry ? "可查看" : "资产不可用"
          : `${formatCount(source.sparsePointCount)} 点`,
      available: source.hasSparseGeometry,
      unavailableReason: "此 job 未发布 SfM 稀疏点云。"
    },
    {
      id: "gaussian",
      label: "高斯结果",
      value:
        source.gaussianCount === undefined
          ? source.hasGaussian ? "可查看" : "资产不可用"
          : `${formatCount(source.gaussianCount)} GS`,
      available: source.hasGaussian,
      unavailableReason: "此 job 未发布 Gaussian Splat。"
    }
  ];
}

function formatCount(value: number | undefined): string {
  return value === undefined ? "-" : value.toLocaleString("zh-CN");
}
