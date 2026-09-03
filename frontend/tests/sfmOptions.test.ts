import assert from "node:assert/strict";
import test from "node:test";

import {
  formatSfmGeometricVerification,
  formatSfmPairing,
  isSfmGeometricVerificationAvailable,
  isSfmPairingAvailable,
  type SfmPairing
} from "../src/sfmOptions.ts";
import type { ExperimentalOptionStatus } from "../src/backendOptions.ts";

function status(
  id: SfmPairing,
  available: boolean,
  supportedModes: string[] = []
): ExperimentalOptionStatus<SfmPairing> {
  return {
    id,
    label: id,
    available,
    reason: null,
    experimental: id !== "exhaustive",
    supported_modes: supportedModes
  };
}

test("legacy capability payload permits only the default pairing", () => {
  assert.equal(isSfmPairingAvailable("exhaustive", undefined, "multi_image"), true);
  assert.equal(isSfmPairingAvailable("sequential_loop", undefined, "video"), false);
  assert.equal(isSfmPairingAvailable("vocab_tree", undefined, "multi_image"), false);
});

test("pairing availability requires backend support for the current mode", () => {
  const sequential = status("sequential_loop", true, ["video"]);
  const vocabulary = status("vocab_tree", true, ["multi_image"]);

  assert.equal(isSfmPairingAvailable("sequential_loop", sequential, "video"), true);
  assert.equal(
    isSfmPairingAvailable("sequential_loop", sequential, "multi_image"),
    false
  );
  assert.equal(isSfmPairingAvailable("vocab_tree", vocabulary, "multi_image"), true);
  assert.equal(isSfmPairingAvailable("vocab_tree", vocabulary, "video"), false);
});

test("explicit backend unavailability always wins", () => {
  assert.equal(
    isSfmPairingAvailable(
      "exhaustive",
      status("exhaustive", false, ["multi_image", "video"]),
      "multi_image"
    ),
    false
  );
});

test("historical sequential pairing keeps its compatibility label", () => {
  assert.equal(formatSfmPairing("sequential"), "Sequential（历史）");
});

test("legacy capability permits default geometric verification only", () => {
  assert.equal(
    isSfmGeometricVerificationAvailable("default_v1", undefined),
    true
  );
  assert.equal(isSfmGeometricVerificationAvailable("guided_v1", undefined), false);
  assert.equal(formatSfmGeometricVerification(undefined), "Default v1（默认）");
});

test("guided verification requires explicit backend availability", () => {
  const available = {
    id: "guided_v1" as const,
    label: "Guided v1",
    available: true,
    reason: null,
    experimental: true
  };
  assert.equal(
    isSfmGeometricVerificationAvailable("guided_v1", available),
    true
  );
  assert.equal(
    isSfmGeometricVerificationAvailable("guided_v1", {
      ...available,
      available: false
    }),
    false
  );
});
