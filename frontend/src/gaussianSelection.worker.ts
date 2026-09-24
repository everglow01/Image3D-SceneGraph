import { createP0WorkerHandler, type P0WorkerMessage } from "./localGaussianP0Worker.ts";

const handle = createP0WorkerHandler((reply, transfer = []) => self.postMessage(reply, { transfer }));
self.addEventListener("message", (event: MessageEvent<P0WorkerMessage>) => { void handle(event.data); });
