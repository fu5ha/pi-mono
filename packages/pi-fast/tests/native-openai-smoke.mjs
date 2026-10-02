import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { findPackageJSON } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

// Run against an explicitly selected host without changing monorepo dependencies.
const [hostPath, ...extra] = process.argv.slice(2);
assert.ok(hostPath && extra.length === 0, "Usage: smoke:native-openai -- <pi-coding-agent-package-directory>");
const hostManifestPath = join(resolve(hostPath), "package.json");
const manifest = JSON.parse(await readFile(hostManifestPath, "utf8"));
assert.equal(manifest.name, "@earendil-works/pi-coding-agent");
const aiManifestPath = findPackageJSON("@earendil-works/pi-ai", pathToFileURL(hostManifestPath));
assert.ok(aiManifestPath, "Host has no pi-ai dependency");
const aiManifest = JSON.parse(await readFile(aiManifestPath, "utf8"));
const { InMemoryCredentialStore } = await import(pathToFileURL(resolve(dirname(aiManifestPath), aiManifest.main)).href);
const {
  createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager,
} = await import(pathToFileURL(resolve(hostPath, manifest.main)).href);

process.env.CI = "1";
process.env.PI_OFFLINE = "1";
const dir = await mkdtemp(join(tmpdir(), "pi-fast-native-openai-"));
process.env.PI_CODING_AGENT_DIR = dir;
const requests = [];
const originalFetch = globalThis.fetch;
let session;
try {
  // Block every network request; use only synthetic in-memory credentials.
  globalThis.fetch = async (url, init) => {
    assert.equal(String(url), "https://api.openai.com/v1/responses");
    requests.push(JSON.parse(init.body));
    const item = {
      type: "message", id: "msg_fixture", role: "assistant", status: "completed",
      content: [{ type: "output_text", text: "OK", annotations: [] }],
    };
    const events = [
      { type: "response.output_item.added", output_index: 0, item: { ...item, content: [] } },
      { type: "response.content_part.added", output_index: 0, content_index: 0, part: { type: "output_text", text: "" } },
      { type: "response.output_text.delta", output_index: 0, content_index: 0, delta: "OK" },
      { type: "response.output_item.done", output_index: 0, item },
      { type: "response.completed", response: {
        status: "completed", output: [item], usage: { input_tokens: 10, output_tokens: 1, total_tokens: 11 },
      } },
    ];
    return new Response(events.map(event => `data: ${JSON.stringify(event)}\n\n`).join(""), {
      headers: { "content-type": "text/event-stream" },
    });
  };
  const { default: piFast } = await import("../extensions/index.ts");
  const credentials = new InMemoryCredentialStore();
  await credentials.modify("openai", async () => ({
    type: "oauth", access: "unused-chatgpt-fixture", refresh: "unused-fixture",
    expires: Date.now() + 3_600_000, clientId: "fixture-client",
    scopes: ["chatgpt.tokens.use.direct"],
  }));
  const modelRuntime = await ModelRuntime.create({
    credentials, modelsPath: null, modelsStorePath: join(dir, "models-store.json"),
  });
  assert.ok(modelRuntime.getProvider("openai")?.auth.oauth, "Host needs native OpenAI OAuth (Pi 0.99.1 or later)");
  assert.equal(modelRuntime.isUsingSubscription("openai"), true);
  const model = modelRuntime.getModel("openai", "gpt-6.1-sol");
  assert.ok(model, "Host has no native GPT-6.1 Sol model");
  const settingsManager = SettingsManager.inMemory({
    compaction: { enabled: false }, retry: { enabled: false },
  });
  const errors = [];
  const loader = new DefaultResourceLoader({
    cwd: dir, agentDir: dir, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    systemPromptOverride: () => "Test assistant.", extensionFactories: [piFast],
  });
  await loader.reload();
  const result = await createAgentSession({
    cwd: dir, agentDir: dir, model, modelRuntime, settingsManager, resourceLoader: loader,
    sessionManager: SessionManager.inMemory(dir), thinkingLevel: "low", noTools: true,
  });
  session = result.session;
  assert.deepEqual(result.extensionsResult.errors, []);
  await session.bindExtensions({ mode: "print", onError: error => errors.push(error) });
  await session.prompt("Reply OK");
  assert.equal(requests[0].service_tier, undefined);
  await session.prompt("/fast on");
  await session.prompt("Reply OK again");
  assert.equal(requests[1].model, model.id);
  assert.equal(requests[1].service_tier, "priority");
  assert.equal(session.messages.at(-1).stopReason, "stop");
  await session.prompt("/fast off");
  await session.prompt("Reply OK once more");
  assert.equal(requests[2].service_tier, undefined);
  await session.prompt("/fast on");
  await modelRuntime.setRuntimeApiKey("openai", "sk-unused-api-key-fixture");
  await session.prompt("API-key request");
  assert.equal(requests[3].service_tier, "fast");
  assert.equal(session.messages.at(-1).stopReason, "stop");
  await session.prompt("/fast off");
  await session.prompt("Standard API-key request");
  assert.equal(requests[4].service_tier, undefined);
  await session.prompt("/fast on");
  await modelRuntime.removeRuntimeApiKey("openai");
  await session.prompt("Subscription request again");
  assert.equal(requests[5].service_tier, "priority");
  assert.equal(session.messages.at(-1).stopReason, "stop");
  assert.deepEqual(errors, []);
  console.log(`PASS: Pi ${manifest.version} native OpenAI OAuth/API-key tier selection (mocked HTTP).`);
} finally {
  try {
    session?.dispose();
  } finally {
    globalThis.fetch = originalFetch;
    await rm(dir, { recursive: true, force: true });
  }
}
