import test from "node:test";
import assert from "node:assert/strict";
import {
  findGaussianTrainerStatus,
  formatGaussianTrainerOption
} from "../src/trainerOptions.ts";
import type { GaussianTrainerStatus } from "../src/trainerOptions.ts";

const trainers: GaussianTrainerStatus[] = [
  {
    id: "project",
    label: "Project (gsplat)",
    available: true,
    reason: null,
    setup_command: null,
    revision: "1.5.3",
    license: "Apache-2.0"
  },
  {
    id: "graphdeco",
    label: "Graphdeco official",
    available: false,
    reason: "environment missing",
    setup_command: "setup graphdeco",
    revision: "pinned",
    license: "research"
  }
];

test("trainer options expose availability and preserve ids", () => {
  assert.equal(formatGaussianTrainerOption(trainers[0]), "Project (gsplat)");
  assert.equal(formatGaussianTrainerOption(trainers[1]), "Graphdeco official (unavailable)");
  assert.equal(findGaussianTrainerStatus(trainers, "graphdeco")?.reason, "environment missing");
  assert.equal(findGaussianTrainerStatus(trainers, "project")?.available, true);
});
