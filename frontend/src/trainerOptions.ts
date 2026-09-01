export type GaussianTrainer = "project" | "graphdeco" | "mcmc";

export type GaussianTrainerStatus = {
  id: GaussianTrainer;
  label: string;
  available: boolean;
  reason: string | null;
  setup_command: string | null;
  revision: string;
  license: string;
};

export function formatGaussianTrainerOption(trainer: GaussianTrainerStatus): string {
  const label =
    trainer.id === "project"
      ? "Project v7（gsplat 高斯栅格化）"
      : trainer.id === "mcmc"
        ? "MCMC v1（实验，gsplat）"
        : "Graphdeco 官方训练器（研究与评估）";
  return trainer.available ? label : `${label}（不可用）`;
}

export function findGaussianTrainerStatus(
  trainers: GaussianTrainerStatus[],
  trainerId: GaussianTrainer
): GaussianTrainerStatus | undefined {
  return trainers.find((trainer) => trainer.id === trainerId);
}
