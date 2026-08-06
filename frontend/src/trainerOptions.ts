export type GaussianTrainer = "project" | "graphdeco" | "nerfstudio";

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
  return trainer.available ? trainer.label : `${trainer.label} (unavailable)`;
}

export function findGaussianTrainerStatus(
  trainers: GaussianTrainerStatus[],
  trainerId: GaussianTrainer
): GaussianTrainerStatus | undefined {
  return trainers.find((trainer) => trainer.id === trainerId);
}
