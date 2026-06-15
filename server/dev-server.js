import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { createReadStream, existsSync, readFileSync } from "node:fs";
import { access, mkdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const WEB_DIR = path.join(ROOT, "web");

loadDotEnv(path.join(ROOT, ".env"));

const HOST = process.env.HOST || "127.0.0.1";
const PORT = Number(process.env.PORT || 8080);
const PORT_FALLBACK_ATTEMPTS = Number(process.env.PORT_FALLBACK_ATTEMPTS || 10);
const PYTHON_BIN = path.resolve(ROOT, process.env.PYTHON_BIN || ".venv/bin/python");
const DEMO_DIR = process.env.CULVERT_DEMO_DIR || "data/ulster_demo";
const DATA_DIR = path.join(WEB_DIR, "data");
const OBSERVATIONS_PATH = path.join(ROOT, "data", "processed", "field_observations.geojson");
const OBSERVATION_STATUSES = new Set(["confirmed_culvert", "no_culvert", "uncertain"]);
const MIME_TYPES = new Map([
  [".html", "text/html; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".geojson", "application/geo+json; charset=utf-8"],
  [".png", "image/png"],
  [".jpg", "image/jpeg"],
  [".jpeg", "image/jpeg"],
  [".svg", "image/svg+xml"],
]);

let activeTask = null;

async function handleRequest(request, response) {
  try {
    const url = new URL(request.url || "/", `http://${request.headers.host || "localhost"}`);

    if (url.pathname === "/api/health" && request.method === "GET") {
      await sendJson(response, await healthPayload());
      return;
    }

    if (url.pathname === "/api/findings" && request.method === "GET") {
      await sendFile(response, path.join(DATA_DIR, "findings.geojson"));
      return;
    }

    if (url.pathname === "/api/summary" && request.method === "GET") {
      await sendFile(response, path.join(DATA_DIR, "summary.json"));
      return;
    }

    if (url.pathname === "/api/observations" && request.method === "GET") {
      await sendJson(response, await readObservations());
      return;
    }

    if (url.pathname === "/api/observations" && request.method === "POST") {
      try {
        const payload = await readJsonBody(request);
        const feature = await appendObservation(payload);
        await sendJson(response, { status: "saved", feature }, 201);
      } catch (error) {
        await sendJson(response, { error: error.message || "Invalid observation" }, 400);
      }
      return;
    }

    if (url.pathname === "/api/run-demo" && request.method === "POST") {
      await runExclusiveTask(response, "demo", [
        [PYTHON_BIN, ["-m", "culvert_ai.cli", "run-demo", "--output-dir", DEMO_DIR]],
        [
          PYTHON_BIN,
          [
            "-m",
            "culvert_ai.cli",
            "export-web",
            "--predictions",
            path.join(DEMO_DIR, "processed", "unlabeled_predictions.gpkg"),
            "--output-dir",
            "web/data",
          ],
        ],
      ]);
      return;
    }

    if (url.pathname === "/api/run-actual" && request.method === "POST") {
      await runExclusiveTask(response, "actual-ulster-census", [
        ["bash", ["scripts/run_actual_ulster_census_pipeline.sh"]],
      ]);
      return;
    }

    if (url.pathname === "/api/run-ulster" && request.method === "POST") {
      await runExclusiveTask(response, "ulster-unlabeled", [
        ["bash", ["scripts/run_ulster_unlabeled_pipeline.sh"]],
      ]);
      return;
    }

    if (url.pathname.startsWith("/api/")) {
      await sendJson(response, { error: "Not found" }, 404);
      return;
    }

    const filePath = url.pathname === "/" ? path.join(WEB_DIR, "index.html") : safeWebPath(url.pathname);
    await sendFile(response, filePath);
  } catch (error) {
    console.error(error);
    await sendJson(response, { error: error.message || "Internal server error" }, 500);
  }
}

listenWithFallback(PORT);

function loadDotEnv(filePath) {
  if (!existsSync(filePath)) return;

  const lines = readFileSync(filePath, "utf8").split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#") || !trimmed.includes("=")) continue;

    const [key, ...valueParts] = trimmed.split("=");
    if (!process.env[key]) {
      process.env[key] = valueParts.join("=").replace(/^['"]|['"]$/g, "");
    }
  }
}

function listenWithFallback(port, attempt = 0) {
  const server = createServer(handleRequest);
  const onError = (error) => {
    if (error.code === "EADDRINUSE" && attempt < PORT_FALLBACK_ATTEMPTS) {
      const nextPort = port + 1;
      console.warn(`Port ${port} is already in use. Trying ${nextPort}.`);
      server.close(() => listenWithFallback(nextPort, attempt + 1));
      return;
    }

    throw error;
  };

  server.once("error", onError);
  server.listen(port, HOST, () => {
    server.off("error", onError);
    console.log(`Culvert AI dev server running at http://${HOST}:${port}`);
    console.log("Frontend: /");
    console.log("Backend:  /api/health, /api/findings, /api/summary, /api/run-actual");
  });
}

function safeWebPath(pathname) {
  const decoded = decodeURIComponent(pathname);
  const normalized = path.normalize(decoded).replace(/^(\.\.[/\\])+/, "");
  const filePath = path.join(WEB_DIR, normalized);

  if (!filePath.startsWith(WEB_DIR)) {
    throw new Error("Invalid path");
  }

  return filePath;
}

async function healthPayload() {
  return {
    status: "ok",
    activeTask,
    python: {
      bin: path.relative(ROOT, PYTHON_BIN),
      ready: await canRead(PYTHON_BIN),
    },
    data: {
      findings: await fileInfo(path.join(DATA_DIR, "findings.geojson")),
      summary: await fileInfo(path.join(DATA_DIR, "summary.json")),
      observations: await fileInfo(OBSERVATIONS_PATH),
    },
    commands: {
      demo: "npm run demo",
      dev: "npm run dev",
      tests: "npm test",
      actual: "npm run predict:actual",
      ulsterPipeline: "npm run pipeline:ulster",
    },
  };
}

async function fileInfo(filePath) {
  try {
    const metadata = await stat(filePath);
    return {
      exists: true,
      bytes: metadata.size,
      updatedAt: metadata.mtime.toISOString(),
    };
  } catch {
    return { exists: false };
  }
}

async function canRead(filePath) {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readObservations() {
  try {
    const raw = await readFile(OBSERVATIONS_PATH, "utf8");
    const parsed = JSON.parse(raw);
    if (parsed?.type === "FeatureCollection" && Array.isArray(parsed.features)) {
      return parsed;
    }
  } catch (error) {
    if (error.code !== "ENOENT") {
      throw error;
    }
  }

  return emptyFeatureCollection();
}

async function appendObservation(payload) {
  const feature = observationFeature(payload);
  const collection = await readObservations();
  const existingIndex = collection.features.findIndex(
    (item) => item.properties?.observation_id === feature.properties.observation_id,
  );

  if (existingIndex >= 0) {
    collection.features[existingIndex] = feature;
  } else {
    collection.features.push(feature);
  }

  await mkdir(path.dirname(OBSERVATIONS_PATH), { recursive: true });
  await writeFile(OBSERVATIONS_PATH, `${JSON.stringify(collection, null, 2)}\n`);
  return feature;
}

function emptyFeatureCollection() {
  return {
    type: "FeatureCollection",
    features: [],
  };
}

function observationFeature(payload) {
  const latitude = Number(payload.latitude);
  const longitude = Number(payload.longitude);
  if (!Number.isFinite(latitude) || latitude < -90 || latitude > 90) {
    throw new Error("Observation needs a valid latitude.");
  }
  if (!Number.isFinite(longitude) || longitude < -180 || longitude > 180) {
    throw new Error("Observation needs a valid longitude.");
  }

  const status = OBSERVATION_STATUSES.has(payload.status) ? payload.status : "uncertain";
  const observedAt = safeString(payload.observed_at, 48) || new Date().toISOString();

  return {
    type: "Feature",
    properties: {
      observation_id: safeString(payload.observation_id, 80) || makeObservationId(),
      observed_at: observedAt,
      status,
      candidate_id: safeString(payload.candidate_id, 120),
      road_name: safeString(payload.road_name, 180),
      stream_name: safeString(payload.stream_name, 180),
      notes: safeString(payload.notes, 2000),
      source: safeString(payload.source, 80) || "field_review",
      prediction_score: numberOrNull(payload.prediction_score),
      priority_rank: numberOrNull(payload.priority_rank),
      priority_bucket: safeString(payload.priority_bucket, 40),
      latitude,
      longitude,
    },
    geometry: {
      type: "Point",
      coordinates: [longitude, latitude],
    },
  };
}

async function readJsonBody(request) {
  let body = "";
  for await (const chunk of request) {
    body += chunk;
    if (body.length > 1_000_000) {
      throw new Error("Observation payload is too large.");
    }
  }

  return body ? JSON.parse(body) : {};
}

function safeString(value, maxLength) {
  return String(value || "").trim().slice(0, maxLength);
}

function numberOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function makeObservationId() {
  return `obs_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

async function runExclusiveTask(response, taskName, commands) {
  if (activeTask) {
    await sendJson(response, { error: `Task already running: ${activeTask}` }, 409);
    return;
  }

  activeTask = taskName;
  const startedAt = new Date().toISOString();

  try {
    const logs = [];
    for (const [command, args] of commands) {
      logs.push(await runCommand(command, args));
    }

    await sendJson(response, {
      task: taskName,
      status: "complete",
      startedAt,
      finishedAt: new Date().toISOString(),
      logs,
    });
  } finally {
    activeTask = null;
  }
}

function runCommand(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: ROOT,
      env: {
        ...process.env,
        PATH: `${path.join(ROOT, ".venv", "bin")}:${process.env.PATH || ""}`,
        PYTHONPATH: path.join(ROOT, "src"),
      },
      shell: false,
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", reject);
    child.on("close", (code) => {
      const result = {
        command: [command, ...args].join(" "),
        code,
        stdout: stdout.slice(-12000),
        stderr: stderr.slice(-12000),
      };

      if (code === 0) {
        resolve(result);
      } else {
        const error = new Error(`${command} exited with code ${code}`);
        error.result = result;
        reject(error);
      }
    });
  });
}

async function sendFile(response, filePath) {
  let fileStat;
  try {
    fileStat = await stat(filePath);
  } catch {
    await sendJson(response, { error: "File not found" }, 404);
    return;
  }

  if (!fileStat.isFile()) {
    await sendJson(response, { error: "File not found" }, 404);
    return;
  }

  const extension = path.extname(filePath).toLowerCase();
  response.writeHead(200, {
    "content-type": MIME_TYPES.get(extension) || "application/octet-stream",
    "content-length": fileStat.size,
    "cache-control": "no-store",
  });
  createReadStream(filePath).pipe(response);
}

async function sendJson(response, payload, status = 200) {
  const body = JSON.stringify(payload, null, 2);
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
    "cache-control": "no-store",
  });
  response.end(body);
}
