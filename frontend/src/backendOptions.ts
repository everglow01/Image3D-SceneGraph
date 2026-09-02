export type ExperimentalOptionStatus<T extends string> = {
  id: T;
  label: string;
  available: boolean;
  reason: string | null;
  experimental: boolean;
  supported_modes?: string[];
  setup_command?: string | null;
};

type BackendAvailability = { available: boolean };
type BackendOutputs = { supported_outputs: string[] };

export function isBackendAvailable(
  backendId: string,
  statuses: Record<string, BackendAvailability> | null
) {
  return statuses?.[backendId]?.available ?? true;
}

export function isOutputSupported(
  outputType: string,
  backendStatus: BackendOutputs | undefined
) {
  return backendStatus?.supported_outputs.includes(outputType) ?? true;
}
