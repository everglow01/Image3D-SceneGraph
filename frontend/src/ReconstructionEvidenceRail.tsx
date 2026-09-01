import type { EvidenceStage, EvidenceStageId } from "./reconstructionEvidence";

type Props = {
  stages: EvidenceStage[];
  active: EvidenceStageId | null;
  onSelect: (stage: EvidenceStageId) => void;
};

export function ReconstructionEvidenceRail({ stages, active, onSelect }: Props) {
  return (
    <nav className="evidence-rail" aria-label="重建证据轨道">
      {stages.map((stage, index) => (
        <div className="evidence-stage-wrap" key={stage.id}>
          {index > 0 && <span className="evidence-connector" aria-hidden="true">→</span>}
          <button
            aria-current={active === stage.id ? "step" : undefined}
            className={`evidence-stage ${active === stage.id ? "active" : ""}`}
            disabled={!stage.available}
            onClick={() => onSelect(stage.id)}
            title={stage.available ? `查看${stage.label}` : stage.unavailableReason}
            type="button"
          >
            <span className="evidence-stage-index">{String(index + 1).padStart(2, "0")}</span>
            <span className="evidence-stage-copy">
              <strong>{stage.label}</strong>
              <small>{stage.value}</small>
            </span>
            <i className={stage.available ? "available" : "unavailable"} aria-hidden="true" />
          </button>
        </div>
      ))}
    </nav>
  );
}
