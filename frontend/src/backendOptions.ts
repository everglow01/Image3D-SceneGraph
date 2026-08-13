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
