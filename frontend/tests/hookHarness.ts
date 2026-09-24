export function findNode(node: any, predicate: (node: any) => boolean): any {
  if (!node || typeof node !== "object") return null;
  if (predicate(node)) return node;
  for (const child of [node.props?.children].flat(Infinity)) {
    const result = findNode(child, predicate);
    if (result) return result;
  }
  return null;
}

export function createHookHarness() {
  const hooks: any[] = [], effects: (() => void)[] = [];
  let cursor = 0, dirty = true;
  const react = {
    useRef(initial: unknown) {
      const index = cursor++;
      return hooks[index] ??= { current: initial };
    },
    useState(initial: unknown) {
      const index = cursor++;
      if (!(index in hooks)) hooks[index] = initial;
      return [hooks[index], (next: any) => {
        const value = typeof next === "function" ? next(hooks[index]) : next;
        if (!Object.is(value, hooks[index])) { hooks[index] = value; dirty = true; }
      }];
    },
    useEffect(callback: () => (() => void) | void, deps: unknown[]) {
      const index = cursor++, previous = hooks[index];
      if (!previous || deps.length !== previous.deps.length || deps.some((d, i) => !Object.is(d, previous.deps[i]))) {
        effects.push(() => {
          previous?.cleanup?.();
          hooks[index] = { deps, cleanup: callback() };
        });
      }
    }
  };
  return {
    react,
    invalidate() { dirty = true; },
    unmount() { for (const hook of hooks) hook?.cleanup?.(); },
    async flush(render: () => void, until?: () => boolean) {
      // A bounded microtask drain is not a real React scheduler or a timer/IO flush.
      for (let i = 0; i < 100; i++) {
        await Promise.resolve();
        if (dirty) {
          dirty = false; cursor = 0; render();
          while (effects.length) effects.shift()!();
        }
        if (until?.()) return;
      }
      if (until) throw new Error("Component did not reach the expected state within 100 microtask turns");
    }
  };
}
