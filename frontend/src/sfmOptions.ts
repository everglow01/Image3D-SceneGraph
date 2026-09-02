import type { ExperimentalOptionStatus } from "./backendOptions";

export type SfmFeatureProfile = "sift_v1" | "aliked_n16rot_v1";
export type SfmLocalMatcher = "bruteforce" | "lightglue";
export type SfmPairing = "exhaustive" | "sequential_loop" | "vocab_tree";

export type SfmLocalMatcherStatus = ExperimentalOptionStatus<SfmLocalMatcher> & {
  pairings?: ExperimentalOptionStatus<SfmPairing>[];
};

export type SfmFeatureStatus = ExperimentalOptionStatus<SfmFeatureProfile> & {
  local_matchers?: SfmLocalMatcherStatus[];
};

export const sfmFeatureOptions: Array<{
  id: SfmFeatureProfile;
  label: string;
}> = [
  { id: "sift_v1", label: "SIFT v1（默认）" },
  { id: "aliked_n16rot_v1", label: "ALIKED N16Rot v1（实验）" }
];

export const sfmLocalMatcherOptions: Array<{
  id: SfmLocalMatcher;
  label: string;
}> = [
  { id: "bruteforce", label: "Brute-force（默认）" },
  { id: "lightglue", label: "LightGlue（实验）" }
];

export const sfmPairingOptions: Array<{ id: SfmPairing; label: string }> = [
  { id: "exhaustive", label: "Exhaustive（默认）" },
  { id: "sequential_loop", label: "Sequential + Loop（实验）" },
  { id: "vocab_tree", label: "Vocab Tree（实验）" }
];

export function isSfmPairingModeSupported(
  status: ExperimentalOptionStatus<SfmPairing> | undefined,
  mode: string
): boolean {
  return status?.supported_modes?.includes(mode) ?? false;
}

export function isSfmPairingAvailable(
  pairing: SfmPairing,
  status: ExperimentalOptionStatus<SfmPairing> | undefined,
  mode: string
): boolean {
  if (
    status?.available === false ||
    (pairing !== "exhaustive" && status?.available !== true)
  ) {
    return false;
  }
  return status === undefined || isSfmPairingModeSupported(status, mode);
}

export function formatSfmFeatureProfile(value: string | undefined): string {
  const profile = value ?? "sift_v1";
  return (
    sfmFeatureOptions.find((option) => option.id === profile)?.label ??
    `未知特征（${profile}）`
  );
}

export function formatSfmLocalMatcher(value: string | undefined): string {
  const matcher = value ?? "bruteforce";
  return (
    sfmLocalMatcherOptions.find((option) => option.id === matcher)?.label ??
    `未知局部匹配器（${matcher}）`
  );
}

export function formatSfmPairing(value: string | undefined): string {
  if (value === "sequential") return "Sequential（历史）";
  const pairing = value ?? "exhaustive";
  return (
    sfmPairingOptions.find((option) => option.id === pairing)?.label ??
    `未知图像对策略（${pairing}）`
  );
}
