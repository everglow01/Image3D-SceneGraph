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
    label: "Project v7 (gsplat)",
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
  },
  {
    id: "mcmc",
    label: "MCMC v1",
    available: true,
    reason: null,
    setup_command: null,
    revision: "gsplat-1.5.3-mcmc-v1",
    license: "Apache-2.0"
  }
];

test("trainer options expose availability and preserve ids", () => {
  assert.equal(formatGaussianTrainerOption(trainers[0]), "Project v7（gsplat 高斯栅格化）");
  assert.equal(formatGaussianTrainerOption(trainers[1]), "Graphdeco 官方训练器（研究与评估）（不可用）");
  assert.equal(findGaussianTrainerStatus(trainers, "graphdeco")?.reason, "environment missing");
  assert.equal(findGaussianTrainerStatus(trainers, "project")?.available, true);
  assert.equal(formatGaussianTrainerOption(trainers[2]), "MCMC v1（实验，gsplat）");
});
