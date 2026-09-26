import type { RefObject } from "react";

export type ViewerLeaveRef = RefObject<(() => Promise<boolean>) | null>;
export type LocalEditorHandle = { leave(): Promise<boolean>; dispose(): Promise<void> };

// App source changes and in-viewer renderer/mode changes use the same live gate.
export async function leaveGaussianViewer(gate?: ViewerLeaveRef): Promise<boolean> {
  try { return await gate?.current?.() ?? true; }
  catch (error) {
    window.alert(error instanceof Error ? error.message : "编辑器释放失败，未切换；请保留草稿后重试");
    return false;
  }
}
