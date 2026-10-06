import assert from "node:assert/strict";
import test from "node:test";
import { readFile, access } from "node:fs/promises";
import { join } from "node:path";
import { Type } from "typebox";
import { createEditToolDefinition, createWriteToolDefinition } from "@earendil-works/pi-coding-agent";
import { codexHarness, textResponse } from "../../../tests/codex-harness.mjs";

process.env.CI = "1";

const { default: piCodexTools } = await import("../extensions/index.ts");
const { supportsOpenAIGrammarTools } = await import("../src/model-support.ts");

function makePi(initialActive = ["read", "write", "edit", "bash"]) {
  const handlers = new Map();
  const tools = new Map();
  let active = [...initialActive];
  return {
    handlers,
    tools,
    on(event, handler) {
      const eventHandlers = handlers.get(event) ?? [];
      eventHandlers.push(handler);
      handlers.set(event, eventHandlers);
    },
    registerTool(tool) {
      tools.set(tool.name, tool);
      if ((tool.exposure ?? "direct") === "direct" || tool.exposure === "model-only") {
        if (!active.includes(tool.name)) active.push(tool.name);
      }
    },
    getActiveTools() {
      return [...active];
    },
    getAllTools() {
      return [
        ...["edit", "write"].filter((name) => !tools.has(name)).map((name) => ({
          name, exposure: "direct", sourceInfo: { path: `builtin:${name}` },
        })),
        ...[...tools.values()].map((tool) => ({
          ...tool, sourceInfo: { path: "<fixture-extension>" },
        })),
      ];
    },
    setActiveTools(names) {
      active = [...names];
    },
  };
}

async function applyProviderRequestHooks(pi, payload, context) {
  let nextPayload = payload;
  for (const handler of pi.handlers.get("before_provider_request") ?? []) {
    const result = await handler({ payload: nextPayload }, context);
    if (result !== undefined) nextPayload = result;
  }
  return nextPayload;
}

const codexModel = {
  provider: "openai-codex",
  api: "openai-codex-responses",
  id: "gpt-5.5",
  compat: { supportsOpenAIGrammarTools: true },
};

const ordinaryModel = {
  provider: "anthropic",
  api: "anthropic-messages",
  id: "claude-sonnet",
  compat: {},
};

test("requires both a Responses API and the advertised grammar capability", () => {
  assert.equal(supportsOpenAIGrammarTools(codexModel), true);
  assert.equal(supportsOpenAIGrammarTools({ ...codexModel, api: "openai-completions" }), false);
  assert.equal(supportsOpenAIGrammarTools({ ...codexModel, compat: {} }), false);
  assert.equal(supportsOpenAIGrammarTools(ordinaryModel), false);
});

function unsupportedModel(model) {
  return { ...model, id: "fixture-unsupported", compat: { ...model.compat, supportsOpenAIGrammarTools: false } };
}

function nestedProbe(run) {
  return (pi) => pi.registerTool({
    name: "nested_probe", label: "Probe", description: "Exercise nested editing.",
    parameters: Type.Object({}),
    async execute(_id, _args, _signal, _update, ctx) {
      await run(ctx);
      return { content: [{ type: "text", text: "OK" }], details: undefined };
    },
  });
}

async function runProbe(t, harness) {
  const item = { type: "function_call", id: "fc_probe", call_id: "call_probe", name: "nested_probe", arguments: "{}" };
  let requests = 0;
  t.mock.method(globalThis, "fetch", async () => {
    if (requests++ > 0) return textResponse();
    return new Response([
      { type: "response.output_item.added", output_index: 0, item: { ...item, arguments: "" } },
      { type: "response.function_call_arguments.delta", output_index: 0, delta: "{}" },
      { type: "response.output_item.done", output_index: 0, item },
      { type: "response.completed", response: { status: "completed", output: [item], usage: { input_tokens: 10, output_tokens: 1 } } },
    ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""), {
      headers: { "content-type": "text/event-stream" },
    });
  });
  await harness.session.prompt("Run nested_probe.");
  const result = harness.session.messages.findLast((message) => message.role === "toolResult");
  assert.ok(result, "the model must execute the probe");
  assert.equal(result.isError, false, JSON.stringify(result.content));
  assert.deepEqual(harness.errors, []);
  t.mock.restoreAll();
}

test("replaces edit and write while preserving unrelated active tools", async () => {
  const pi = makePi();
  piCodexTools(pi);
  const context = { model: codexModel };

  await pi.handlers.get("session_start")[0]({}, context);
  assert.deepEqual(pi.getActiveTools(), ["read", "bash", "apply_patch"]);
  assert.equal(pi.tools.get("edit").exposure, "codemode");
  assert.equal(pi.tools.get("write").exposure, "codemode");

  assert.equal(pi.tools.get("apply_patch").executionMode, "sequential");
  assert.deepEqual(
    await applyProviderRequestHooks(pi, { model: codexModel.id, parallel_tool_calls: true }, context),
    { model: codexModel.id, parallel_tool_calls: true },
  );

  await pi.handlers.get("model_select")[0]({}, { model: ordinaryModel });
  assert.deepEqual(pi.getActiveTools(), ["read", "bash", "edit", "write"]);
  assert.equal(pi.tools.get("edit").exposure, "direct");
  assert.equal(pi.tools.get("write").exposure, "direct");
});

test("rejects apply_patch execution for unsupported models", async () => {
  const pi = makePi();
  piCodexTools(pi);
  const tool = pi.tools.get("apply_patch");
  await assert.rejects(
    tool.execute("call", { patch: "*** Begin Patch\n*** End Patch" }, undefined, undefined, { model: ordinaryModel, cwd: process.cwd() }),
    /only available for OpenAI models that advertise grammar-tool support/,
  );
});

test("restores only file tools that were active before replacement", async () => {
  const pi = makePi(["read", "edit", "bash"]);
  piCodexTools(pi);
  const context = { model: codexModel };

  await pi.handlers.get("session_start")[0]({}, context);
  assert.deepEqual(pi.getActiveTools(), ["read", "bash", "apply_patch"]);

  await pi.handlers.get("model_select")[0]({}, { model: ordinaryModel });
  assert.deepEqual(pi.getActiveTools(), ["read", "bash", "edit"]);
});

test("activates GPT-6 Astra based on capability, without a native filesystem binding", async () => {
  const pi = makePi();
  piCodexTools(pi);

  for (const api of ["openai-responses", "openai-codex-responses"]) {
    const model = { ...codexModel, id: "gpt-6-astra", api };
    await pi.handlers.get("session_start")[0]({}, { model });
    assert.deepEqual(pi.getActiveTools(), ["read", "bash", "apply_patch"]);
    await pi.handlers.get("model_select")[0]({}, { model: { ...model, compat: {} } });
    assert.deepEqual(pi.getActiveTools(), ["read", "bash", "edit", "write"]);
  }
});

test("exposes apply_patch as a raw grammar tool", () => {
  const pi = makePi();
  piCodexTools(pi);
  const tool = pi.tools.get("apply_patch");
  assert.equal(tool.constrainedSampling.type, "grammar");
  assert.equal(tool.exposure, "model-only");
  assert.match(tool.constrainedSampling.variants.openai_lark, /start: begin_patch hunk\+ end_patch/);
  assert.match(tool.description, /FREEFORM/);
});

test("does not override native file tools on an unsupported initial model", async () => {
  const pi = makePi();
  piCodexTools(pi);
  await pi.handlers.get("session_start")[0]({}, { model: ordinaryModel });
  assert.equal(pi.tools.has("edit"), false);
  assert.equal(pi.tools.has("write"), false);
  assert.deepEqual(pi.getActiveTools(), ["read", "write", "edit", "bash"]);
});

test("real Pi loadout exposes native editing only through nested calls while apply_patch is active", async () => {
  const harness = await codexHarness([piCodexTools]);
  try {
    const { session, model } = harness;
    assert.ok(session.getActiveToolNames().includes("apply_patch"));
    assert.ok(!session.getCallableToolNames().includes("apply_patch"));
    for (const name of ["edit", "write"]) {
      assert.ok(!session.getActiveToolNames().includes(name));
      assert.ok(session.getCallableToolNames().includes(name));
      assert.equal(session.getToolDefinition(name).exposure, "codemode");
    }
    await session.setModel({
      ...model,
      id: "fixture-unsupported",
      compat: { ...model.compat, supportsOpenAIGrammarTools: false },
    });
    assert.ok(!session.getActiveToolNames().includes("apply_patch"));
    for (const name of ["edit", "write"]) {
      assert.ok(session.getActiveToolNames().includes(name));
      assert.equal(session.getToolDefinition(name).exposure, "direct");
    }
    await session.setModel(model);
    for (const name of ["edit", "write"]) {
      assert.ok(!session.getActiveToolNames().includes(name));
      assert.ok(session.getCallableToolNames().includes(name));
    }
  } finally {
    await harness.close();
  }
});

test("nested write and edit execute against a disposable file; apply_patch rejects nested calls", async (t) => {
  const harness = await codexHarness([piCodexTools, nestedProbe(async (ctx) => {
    assert.equal((await ctx.executeTool("write", { path: "nested.txt", content: "before\n" })).isError, false);
    assert.equal((await ctx.executeTool("edit", { path: "nested.txt", edits: [{ oldText: "before", newText: "after" }] })).isError, false);
    assert.equal((await ctx.executeTool("apply_patch", { patch: "*** Begin Patch\n*** Add File: forbidden.txt\n+no\n*** End Patch" })).isError, true);
  })]);
  try {
    await runProbe(t, harness);
    assert.equal(await readFile(join(harness.ctx.cwd, "nested.txt"), "utf8"), "after\n");
    await assert.rejects(access(join(harness.ctx.cwd, "forbidden.txt")));
  } finally {
    await harness.close();
  }
});

for (const selection of ["defaultTools", "tools"]) {
  test(`${selection} excluding edit/write prevents nested editing across model switches`, async (t) => {
    const harness = await codexHarness([piCodexTools, nestedProbe(async (ctx) => {
      assert.equal((await ctx.executeTool("write", { path: "excluded.txt", content: "no\n" })).isError, true);
      assert.equal((await ctx.executeTool("edit", { path: "excluded.txt", edits: [{ oldText: "no", newText: "yes" }] })).isError, true);
    })], { [selection]: ["read", "bash", "nested_probe", "apply_patch"] });
    try {
      for (const model of [harness.model, unsupportedModel(harness.model), harness.model]) {
        await harness.session.setModel(model);
        for (const name of ["edit", "write"]) {
          assert.ok(!harness.session.getActiveToolNames().includes(name));
          assert.ok(!harness.session.getCallableToolNames().includes(name));
        }
        await runProbe(t, harness);
      }
      await assert.rejects(access(join(harness.ctx.cwd, "excluded.txt")));
    } finally {
      await harness.close();
    }
  });
}

for (const approvalFirst of [false, true]) {
  test(`approval overrides survive model switches (${approvalFirst ? "before" : "after"} apply_patch extension)`, async (t) => {
    const calls = [];
    const approval = (pi) => {
      for (const definition of [createEditToolDefinition(process.cwd()), createWriteToolDefinition(process.cwd())]) {
        pi.registerTool({
          ...definition,
          async execute() {
            calls.push(definition.name);
            return { content: [{ type: "text", text: "Approval required" }], details: undefined };
          },
        });
      }
    };
    const extensions = approvalFirst ? [approval, piCodexTools] : [piCodexTools, approval];
    const harness = await codexHarness([...extensions, nestedProbe(async (ctx) => {
      await ctx.executeTool("write", { path: "approval.txt", content: "no\n" });
      await ctx.executeTool("edit", { path: "approval.txt", edits: [{ oldText: "no", newText: "yes" }] });
    })]);
    try {
      for (const model of [harness.model, unsupportedModel(harness.model), harness.model]) {
        await harness.session.setModel(model);
        for (const name of ["edit", "write"]) {
          assert.equal(harness.session.getToolDefinition(name).exposure ?? "direct", "direct");
          assert.ok(harness.session.getActiveToolNames().includes(name));
        }
        await runProbe(t, harness);
      }
      assert.deepEqual(calls, ["write", "edit", "write", "edit", "write", "edit"]);
      await assert.rejects(access(join(harness.ctx.cwd, "approval.txt")));
    } finally {
      await harness.close();
    }
  });
}

test("explicit activation after the snapshot survives supported and unsupported model switches", async () => {
  const harness = await codexHarness([piCodexTools], { defaultTools: ["read", "edit", "bash"] });
  try {
    harness.api.setActiveTools([...harness.session.getActiveToolNames(), "edit", "write"]);
    await harness.session.setModel({ ...harness.model, id: "another-supported-model" });
    for (const name of ["edit", "write"]) assert.ok(harness.session.getActiveToolNames().includes(name));
    await harness.session.setModel(unsupportedModel(harness.model));
    for (const name of ["edit", "write"]) {
      assert.ok(harness.session.getActiveToolNames().includes(name));
      assert.equal(harness.session.getToolDefinition(name).exposure ?? "direct", "direct");
    }
  } finally {
    await harness.close();
  }
});

test("older Pi without exposure metadata uses the legacy loadout without registering native overrides", async () => {
  const pi = makePi(["read", "edit", "bash"]);
  const getAllTools = pi.getAllTools;
  pi.getAllTools = () => getAllTools().map(({ exposure, sourceInfo, ...tool }) => tool);
  piCodexTools(pi);
  await pi.handlers.get("session_start")[0]({}, { model: codexModel });
  assert.deepEqual(pi.getActiveTools(), ["read", "bash", "apply_patch"]);
  for (const name of ["edit", "write"]) assert.ok(!pi.tools.has(name));
  pi.setActiveTools([...pi.getActiveTools(), "write"]);
  await pi.handlers.get("model_select")[0]({}, { model: ordinaryModel });
  assert.deepEqual(pi.getActiveTools(), ["read", "bash", "write", "edit"]);
  for (const name of ["edit", "write"]) assert.ok(!pi.tools.has(name));
});
