#!/usr/bin/env node
/**
 * Download built model artifacts from the shared ShelfSet Google Drive folder.
 *
 * The folder must remain publicly accessible. The script prefers `gdown`
 * (https://github.com/wkentaro/gdown) for a fully automatic download. If
 * `gdown` is not installed it falls back to printing the share link and
 * instructions for a manual download.
 */

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const FOLDER_ID = "1l-iA7Y6t8sXIwislklIE_2iRVqQ_PCW0";
const FOLDER_URL = `https://drive.google.com/drive/folders/${FOLDER_ID}?usp=sharing`;

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = resolve(__dirname, "..");
const OUTPUT_DIR = resolve(PROJECT_ROOT, "artifacts", "downloaded");

function hasCommand(cmd) {
  const result = spawnSync(cmd, ["--version"], {
    shell: true,
    stdio: "ignore",
  });
  return result.status === 0;
}

function runGdown() {
  if (!existsSync(OUTPUT_DIR)) {
    mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  console.log(`Downloading models from ${FOLDER_URL}`);
  console.log(`Saving to ${OUTPUT_DIR}\n`);

  const gdown = spawnSync("gdown", ["--folder", FOLDER_ID, "-O", OUTPUT_DIR], {
    stdio: "inherit",
    shell: true,
  });

  if (gdown.status !== 0) {
    console.error("\ngdown failed. Falling back to manual instructions.\n");
    printManualInstructions();
    process.exit(1);
  }

  console.log("\nModels downloaded successfully.");
}

function printManualInstructions() {
  console.log("Built models folder:");
  console.log(`  ${FOLDER_URL}\n`);
  console.log("To download automatically, install gdown and run this script again:");
  console.log("  pip install gdown");
  console.log("  node scripts/download-models.js\n");
  console.log("Or download the folder contents manually and place them in:");
  console.log(`  ${OUTPUT_DIR}\n`);
}

if (!hasCommand("gdown")) {
  console.warn("gdown is not installed or not on PATH.\n");
  printManualInstructions();
  process.exit(1);
}

runGdown();
