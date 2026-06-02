#!/usr/bin/env node
/**
 * extract-models.js
 *
 * Extracts a fresh models.json from the published `command-code` npm package
 * (specifically from its bundled dist/index.mjs which contains the authoritative
 * model list, provider groups, etc.).
 *
 * This replaces the previous manual / stale fetch from the now-404
 * ninehills/pi-commandcode-provider repo.
 *
 * Usage:
 *   node scripts/extract-models.js
 *   # or make executable + ./scripts/extract-models.js
 *
 * It writes ./models.json (which is gitignored and used as offline fallback
 * by the Rust proxy at startup).
 */

import { execSync } from "child_process";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

function log(...args) {
  console.error("[extract-models]", ...args);
}

function findBalancedBlock(haystack, startNeedle) {
  const idx = haystack.indexOf(startNeedle);
  if (idx === -1) return null;
  let s = haystack.lastIndexOf("{", idx);
  if (s === -1) s = haystack.lastIndexOf("[", idx);
  if (s === -1) return null;

  const opener = haystack[s];
  const closer = opener === "{" ? "}" : "]";
  let depth = 1;
  let i = s + 1;
  let instr = false;
  let sc = "";

  while (i < haystack.length && depth > 0) {
    const ch = haystack[i];
    if (instr) {
      if (ch === sc && haystack[i - 1] !== "\\") instr = false;
    } else {
      if ("\"'".includes(ch)) {
        instr = true;
        sc = ch;
      } else if (ch === opener) {
        depth++;
      } else if (ch === closer) {
        depth--;
      }
    }
    i++;
  }
  return haystack.slice(s, i);
}

/**
 * Find an object literal that is assigned via `NAME={...` or `(NAME={...`
 * This is more reliable than plain lastIndexOf { when the literal appears
 * inside larger expressions (common in minified bundles).
 */
function findAssignedObjectBlock(haystack, name) {
  // Try several common minified assignment patterns
  const patterns = [
    `${name}={`,
    `${name} = {`,
    `(${name}={`,
    `(${name} = {`,
    `=${name}={`,
  ];
  for (const p of patterns) {
    const idx = haystack.indexOf(p);
    if (idx !== -1) {
      // The "{" belonging to this assignment is the first { after the =
      const braceIdx = haystack.indexOf("{", idx);
      if (braceIdx !== -1) {
        const block = findBalancedBlock(haystack, haystack.slice(braceIdx, braceIdx + 20));
        if (block) return block;
      }
    }
  }
  // Fallback to the generic finder
  return findBalancedBlock(haystack, `${name}={`);
}

function extractString(text, field) {
  const re = new RegExp(field + ':"([^"]*)"', "i");
  const m = text.match(re);
  return m ? m[1] : null;
}

function extractBool(text, field) {
  const re = new RegExp(field + ":(!0|true|!1|false)", "i");
  const m = text.match(re);
  if (!m) return null;
  return m[1] === "!0" || m[1] === "true";
}

function extractNum(text, field) {
  const re = new RegExp(field + ":([\\deE.+-]+)", "i");
  const m = text.match(re);
  if (!m) return null;
  const v = parseFloat(m[1]);
  return Number.isFinite(v) ? (v > 1e5 ? Math.round(v) : v) : null;
}

function extractArr(text, field) {
  const re = new RegExp(field + ":\\[([^\\]]*)\\]", "i");
  const m = text.match(re);
  if (!m) return null;
  const items = [];
  const strRe = /"([^"]*)"/g;
  let mm;
  while ((mm = strRe.exec(m[1])) !== null) items.push(mm[1]);
  return items.length ? items : null;
}

function extractModelsJson(mjsContent) {
  // --- Resolve Wt (well-known providers)
  // Use the assignment-aware finder because of minification patterns like (Wt={...
  const wtBlock = findAssignedObjectBlock(mjsContent, "Wt");
  const wt = {};
  if (wtBlock) {
    const pairs = wtBlock.match(/([A-Z_]+):"([^"]+)"/g) || [];
    for (const p of pairs) {
      const m = p.match(/([A-Z_]+):"([^"]+)"/);
      if (m) wt[m[1]] = m[2];
    }
  }
  log("Wt providers:", wt);

  // --- pn (provider groups / catalog)
  // We search for a distinctive entry to locate the block
  let pnBlock = findBalancedBlock(mjsContent, '"command-code":{id:"command-code"');
  if (!pnBlock) {
    pnBlock = findBalancedBlock(mjsContent, "pn={");
  }
  const provider_groups = [];
  const providersMap = {};

  if (pnBlock) {
    // Match group entries. Keys may be unquoted (command-code) or quoted ("github-copilot").
    // We capture starting from either "key" or key (unquoted) then :{id:...
    const groupRe =
      /(?:"([a-z0-9-]+)"|([a-z0-9-]+)):\{id:"([^"]+)",label:"([^"]+)",shortLabel:"([^"]*)",description:"([^"]*)",supportedModelProviders:\[([^\]]*)\]/g;
    let gm;
    while ((gm = groupRe.exec(pnBlock)) !== null) {
      const id = gm[1] || gm[2];
      const label = gm[4];
      const shortLabel = gm[5];
      const description = gm[6];
      const supportedRaw = gm[7];
      const providers = [];
      const provRefs = supportedRaw.match(/([A-Z_]+)/g) || [];
      for (const ref of provRefs) {
        if (wt[ref]) {
          providers.push(wt[ref]);
        }
        // ignore other short vars / junk here; we only care about the known Wt ones for groups
      }
      provider_groups.push({
        id,
        label,
        short_label: shortLabel || id,
        description: description || "",
        providers,
      });
      providersMap[id] = label;
    }
  }
  log("provider_groups:", provider_groups.map((g) => g.id));

  // --- an (the actual model definitions map)
  const anBlock = findBalancedBlock(mjsContent, "SONNET_4_6:{id:");
  if (!anBlock) {
    throw new Error("Could not locate models definition block (an) in dist/index.mjs");
  }

  const models = [];
  // Each entry: KEY:{id:"...",provider:XXX,spec:...,label:"..", ...}
  // We iterate by finding successive KEY:{id: patterns
  const entryStartRe = /([A-Z][A-Z0-9_]*):\s*\{id:"/g;
  let em;
  while ((em = entryStartRe.exec(anBlock)) !== null) {
    const key = em[1];
    const start = em.index + em[0].length - 4; // back up to the { of this model
    // find matching } for this model value (simple depth count from here)
    let d = 1;
    let j = start + 1;
    let ins = false;
    let sc = "";
    while (j < anBlock.length && d > 0) {
      const ch = anBlock[j];
      if (ins) {
        if (ch === sc && anBlock[j - 1] !== "\\") ins = false;
      } else {
        if ("\"'".includes(ch)) {
          ins = true;
          sc = ch;
        } else if (ch === "{") d++;
        else if (ch === "}") d--;
      }
      j++;
    }
    const valText = anBlock.slice(start, j);

    let provRef = (valText.match(/provider:([A-Za-z0-9_.]+)/) || [])[1] || "Qt";
    let provider;
    if (provRef.startsWith("Wt.")) {
      const k = provRef.split(".")[1];
      provider = wt[k] || k.toLowerCase();
    } else {
      // Qt (and any other short var) => these are Command Code hosted / BYOK open models
      provider = "command-code";
    }

    const model = {
      key,
      id: extractString(valText, "id"),
      provider,
      spec: extractString(valText, "spec") || "chatComplete",
      label: extractString(valText, "label"),
      name: extractString(valText, "name"),
      description: extractString(valText, "description"),
      reasoning: !!extractBool(valText, "reasoning"),
      reasoningEfforts: extractArr(valText, "reasoningEfforts"),
      contextWindow: extractNum(valText, "contextWindow") || 0,
      maxOutputTokens: extractNum(valText, "maxOutputTokens") || extractNum(valText, "maxTokens") || 0,
      vendorLabel: extractString(valText, "vendorLabel") || undefined,
    };

    // Clean undefined / empty (Rust side uses #[serde(default)] etc)
    if (model.vendorLabel === undefined) delete model.vendorLabel;
    if (!model.reasoningEfforts) delete model.reasoningEfforts;
    // Always keep contextWindow / maxOutputTokens (even as 0) because ModelDef
    // fields are u64 without #[serde(default)] in the Rust proxy.
    // (the proxy mostly only cares about id + provider/vendorLabel anyway)

    models.push(model);
  }

  log(`Extracted ${models.length} models`);

  // Also make sure the direct Wt providers are in the map (and pretty-ish labels)
  for (const [k, v] of Object.entries(wt)) {
    if (!providersMap[v]) {
      providersMap[v] = k
        .split("_")
        .map((w) => w.charAt(0) + w.slice(1).toLowerCase())
        .join(" ");
    }
  }

  return {
    providers: providersMap,
    provider_groups,
    models,
    pricing: [],
  };
}

async function main() {
  const args = process.argv.slice(2);
  const outArg = args.find((a) => a.startsWith("--out="));
  const outPath = outArg
    ? path.resolve(outArg.split("=")[1])
    : path.join(ROOT, "models.json");

  // 1. Resolve latest version (no extra deps)
  let version;
  try {
    version = execSync("npm view command-code version", {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch (e) {
    log("Failed to resolve latest version via npm view, trying fallback...");
    version = "0.30.2"; // last known at time of writing
  }
  log(`command-code version: ${version}`);

  const tgzName = `command-code-${version}.tgz`;
  const tgzUrl = `https://registry.npmjs.org/command-code/-/${tgzName}`;

  // 2. Download + extract ONLY dist/index.mjs using shell tools (curl + tar present on dev machines)
  let mjsContent;
  try {
    // We stream to tar to avoid writing huge temp tgz if possible, but for simplicity we use a temp file.
    const tmpTgz = `/tmp/${tgzName}`;
    log(`Downloading ${tgzUrl}`);
    execSync(`curl -sL --fail -o ${tmpTgz} ${tgzUrl}`, { stdio: "inherit" });

    log("Extracting package/dist/index.mjs from tarball...");
    mjsContent = execSync(`tar -xOf ${tmpTgz} package/dist/index.mjs`, {
      encoding: "utf8",
      maxBuffer: 20 * 1024 * 1024,
    });
    // best effort cleanup
    try { fs.unlinkSync(tmpTgz); } catch {}
  } catch (err) {
    log("Download/extract failed:", err.message);
    // Fallback: if a local models.json already exists, keep it; otherwise error
    if (fs.existsSync(outPath)) {
      log(`Keeping existing ${outPath}`);
      return;
    }
    throw err;
  }

  // 3. Parse
  const modelsJson = extractModelsJson(mjsContent);

  // 4. Write
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify(modelsJson, null, 2) + "\n", "utf8");
  log(`Wrote ${outPath} (${modelsJson.models.length} models)`);

  // 5. Quick sanity
  const sample = modelsJson.models.slice(0, 3).map((m) => m.id);
  log("Sample models:", sample);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
