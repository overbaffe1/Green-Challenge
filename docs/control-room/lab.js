"use strict";

(() => {
  const el = (id) => document.getElementById(id);
  const data = window.GC_REPLAY;
  if (!data) {
    el("missing").hidden = false;
    return;
  }

  const modes = {
    v10: { label: "1,0× · исходный", pair: "v13" },
    v13: { label: "1,3× · по умолчанию", pair: "v10" },
    "v18-drop": { label: "1,8× · с отсечкой", pair: "v18-keep" },
    "v18-keep": { label: "1,8× · без отсечки", pair: "v18-drop" },
  };
  const series = Object.fromEntries(Object.entries(data.captures).map(([key, run]) => [
    key, { ...run, samples: run.requests.flatMap((request) => request.commands) },
  ]));
  const first = series.v13;
  const { control_dt: dt, commands_per_request: perRequest } = first;
  const maxTick = first.samples.length - 1;
  const period = dt * perRequest;
  const formats = [0, 1, 2, 3, 4].map((digits) => new Intl.NumberFormat("ru-RU", {
    minimumFractionDigits: digits, maximumFractionDigits: digits,
  }));
  const fmt = (value, digits = 0) => formats[digits].format(Object.is(value, -0) ? 0 : value);
  const pad = (n) => String(n).padStart(2, "0");
  const num = (n) => n.toFixed(2);
  const put = (id, value) => { el(id).textContent = value; };
  const svgNS = "http://www.w3.org/2000/svg";
  function svg(name, attrs, text) {
    const node = document.createElementNS(svgNS, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function line(x1, y1, x2, y2, cls) {
    return svg("line", { x1, y1, x2, y2, class: cls });
  }
  function text(x, y, anchor, content) {
    return svg("text", { x, y, "text-anchor": anchor, class: "axis-label" }, content);
  }

  let mode = "v13";
  let joint = 1;
  let tick = 0;
  let frameId = null;
  let clockOrigin, playbackStart;
  let markedTick = null;
  let scopeY;
  const scope = el("scope");
  const seek = el("seek");
  seek.max = maxTick;
  put("end-time", `${fmt(maxTick * dt, 3)} с`);
  put("capture-size", `${first.requests.length} запросов / ${first.samples.length} команд`);

  const scopePlot = { left: 48, right: 464, top: 24, bottom: 186 };
  const sx = (index) => scopePlot.left + (scopePlot.right - scopePlot.left) * index / maxTick;
  function waveform(samples) {
    return samples.map((cmd, i) => `${i ? "L" : "M"}${num(sx(i))} ${num(scopeY(cmd[joint]))}`).join("");
  }
  function drawScope() {
    const { left, right, top, bottom } = scopePlot;
    const vals = Object.values(series).flatMap((run) => run.samples.map((cmd) => cmd[joint]));
    const low = Math.min(0, ...vals);
    const high = Math.max(0, ...vals);
    scopeY = (value) => bottom - ((value - low) / (high - low || 1)) * (bottom - top);
    const grid = [text(left, 16, "start", "рад")];
    for (let i = 0; i <= 4; i++) {
      const value = low + (high - low) * i / 4;
      const y = scopeY(value);
      grid.push(line(left, y, right, y, "grid-line"), text(left - 8, y + 4, "end", fmt(value, 2)));
    }
    for (let s = 0; s <= maxTick * dt; s++) {
      const x = sx(s / dt);
      grid.push(line(x, top, x, bottom, "grid-line"), text(x, 205, "middle", `${fmt(s)} с`));
    }
    el("scope-grid").replaceChildren(...grid);
    el("scope-a").setAttribute("d", waveform(series[mode].samples));
    const paired = modes[mode].pair;
    const compare = el("compare").checked;
    el("scope-b").setAttribute("d", compare ? waveform(series[paired].samples) : "");
    el("legend-b-wrap").hidden = !compare;
    put("legend-a", `A / ${modes[mode].label}`);
    put("legend-b", `B / ${modes[paired].label}`);
  }

  const degrees = 180 / Math.PI;
  const rad = (angle) => angle / degrees;
  function pose([shoulder, elbow]) {
    // Fixed geometry and offsets for an illustration, not the simulator's kinematics.
    const a = -59 + shoulder * degrees;
    const b = 66 + elbow * degrees;
    const x = 177 + 108 * Math.cos(rad(a)) + 116 * Math.cos(rad(a + b));
    const y = 218 + 108 * Math.sin(rad(a)) + 116 * Math.sin(rad(a + b));
    return { a, b, x, y };
  }
  function drawOrbit() {
    const path = series[mode].samples.flatMap((cmd, i) => {
      if (i % 8 && i !== maxTick) return [];
      const { x, y } = pose(cmd);
      return [`${i ? "L" : "M"}${num(x)} ${num(y)}`];
    }).join("");
    el("orbit").setAttribute("d", path);
  }

  const tickButtons = Array.from({ length: perRequest }, (_, i) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = pad(i + 1);
    button.setAttribute("aria-label", `${pad(i + 1)} — такт ответа, плюс ${fmt(i * dt * 1000)} мс`);
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => go(Math.floor(tick / perRequest) * perRequest + i));
    return button;
  });
  el("tick-strip").replaceChildren(...tickButtons);
  const nominalHorizon = Math.max(...Object.values(series).map((r) => r.lead + r.row_step * r.rows_per_plan));
  const scheduleMax = Math.ceil(nominalHorizon * 10) / 10;
  function drawSchedules() {
    const r = series[mode];
    const x = (t) => 64 + (650 - 64) * t / scheduleMax;
    const all = [
      svg("rect", { x: x(0), y: 33, width: x(period) - x(0), height: 62, rx: 2, fill: "#2e4d36" }),
      line(x(0), 59, x(scheduleMax), 59, "grid-line"),
      text(x(0), 22, "start", `ПЛАН / ${r.rows_per_plan} СТРОК`),
    ];
    for (let s = 0; s <= scheduleMax; s += .4) {
      all.push(line(x(s), 78, x(s), 95, "grid-line"), text(x(s), 111, "middle", fmt(s * 1000)));
    }
    all.push(text(663, 111, "end", "мс"));
    for (let i = 1; i <= r.rows_per_plan; i++) {
      const px = x(r.lead + i * r.row_step);
      all.push(svg("line", { x1: px, x2: px, y1: (i % 10 === 0 ? 39 : 46), y2: 71, stroke: "#d8f989" }));
    }
    all.push(svg("path", {
      d: `M${x(r.blend_life)} 32v47`, stroke: "#f4c48b", "stroke-dasharray": "3 4",
    }));
    all.push(text(x(r.blend_life), 22, "middle", "ПРЕДЕЛ В БЛЕНДЕ"));
    el("schedules").replaceChildren(...all);
    put("row-step", fmt(r.row_step * 1000, 2));
    put("horizon", fmt((r.lead + r.row_step * r.rows_per_plan) * 1000, 0));
    put("blend-life", fmt(r.blend_life * 1000));
  }

  function updateFrame(index) {
    tick = index;
    const run = series[mode];
    const cmd = run.samples[tick];
    const packet = Math.floor(tick / perRequest);
    const offset = tick % perRequest;
    const row = run.requests[packet];
    const { a, b, x, y } = pose(cmd);
    el("arm-shoulder").setAttribute("transform", `translate(177 218) rotate(${num(a)})`);
    el("arm-elbow").setAttribute("transform", `translate(108 0) rotate(${num(b)})`);
    el("tool-dot").setAttribute("cx", num(x));
    el("tool-dot").setAttribute("cy", num(y));
    const chartX = num(sx(tick));
    el("scope-head").setAttribute("d", `M${chartX} ${scopePlot.top}V${scopePlot.bottom}`);
    el("scope-dot").setAttribute("cx", chartX);
    el("scope-dot").setAttribute("cy", num(scopeY(cmd[joint])));

    put("shoulder", fmt(cmd[0], 4));
    put("elbow", fmt(cmd[1], 4));
    put("live", row.live);
    put("dropped", row.frozen_dropped);
    put("packet", pad(packet + 1));
    put("tick", pad(offset + 1));
    put("clock", `${fmt(tick * dt, 3)} с`);
    put("zoom-time", `+${fmt(offset * dt * 1000)} мс`);
    seek.value = tick;
    seek.setAttribute("aria-valuetext", `${fmt(tick * dt, 3)} секунды, запрос ${packet + 1}, такт ${offset + 1}`);
    el("prev").disabled = tick === 0;
    el("next").disabled = tick === maxTick;
    if (markedTick !== offset) {
      if (markedTick !== null) tickButtons[markedTick].setAttribute("aria-pressed", "false");
      tickButtons[offset].setAttribute("aria-pressed", "true");
      markedTick = offset;
    }
  }

  function pause() {
    if (frameId !== null) cancelAnimationFrame(frameId);
    frameId = null;
    put("play-text", "Пуск");
    put("play-icon", "▶");
  }
  function go(index) {
    pause();
    updateFrame(Math.max(0, Math.min(maxTick, index)));
  }
  function animate(now) {
    const next = Math.min(maxTick, playbackStart + Math.floor((now - clockOrigin) * .5 / (dt * 1000)));
    if (next !== tick) updateFrame(next);
    if (next === maxTick) pause();
    else frameId = requestAnimationFrame(animate);
  }
  el("play").addEventListener("click", () => {
    if (frameId !== null) return pause();
    if (tick === maxTick) updateFrame(0);
    put("play-text", "Пауза");
    put("play-icon", "Ⅱ");
    playbackStart = tick;
    clockOrigin = performance.now();
    frameId = requestAnimationFrame(animate);
  });
  seek.addEventListener("input", () => go(Number(seek.value)));
  el("prev").addEventListener("click", () => go(tick - 1));
  el("next").addEventListener("click", () => go(tick + 1));
  document.addEventListener("visibilitychange", () => { if (document.hidden) pause(); });
  scope.addEventListener("click", (event) => {
    const point = scope.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const local = point.matrixTransform(scope.getScreenCTM().inverse());
    go(Math.round(maxTick * (local.x - scopePlot.left) / (scopePlot.right - scopePlot.left)));
  });
  el("joint").addEventListener("change", () => {
    joint = Number(el("joint").value);
    drawScope();
    updateFrame(tick);
  });
  el("compare").addEventListener("change", () => {
    drawScope();
    updateFrame(tick);
  });
  function loadMode(key) {
    mode = key;
    drawScope();
    drawSchedules();
    drawOrbit();
    updateFrame(tick);
  }
  document.querySelectorAll('input[name="mode"]').forEach((input) => {
    input.addEventListener("change", () => { if (input.checked) loadMode(input.value); });
  });

  const { raw_names: names, model_to_raw: model, wire_to_raw: wire, delta_mask: deltaMask } = data.patch;
  const picker = el("channel");
  names.forEach((name, i) => picker.add(new Option(`${pad(i)} — ${name}`, i)));
  const raw = names.map((_, i) => i);
  const group = (i) => i < 12 ? "legs" : i < 25 ? "torso" : i < 41 ? "hands" : i < 45 ? "base" : "vel";
  const grids = [["raw-cells", raw], ["model-cells", model], ["wire-cells", wire]].map(([id, indices]) => {
    const cells = indices.map((n, i) => {
      const cell = document.createElement("span");
      cell.className = "cell";
      cell.dataset.group = group(n);
      cell.textContent = pad(i);
      cell.title = `${i} / ${names[n]}`;
      return cell;
    });
    el(id).replaceChildren(...cells);
    return cells;
  });
  function selectChannel(rawIndex) {
    const m = model.indexOf(rawIndex);
    const w = wire.indexOf(rawIndex);
    picker.value = rawIndex;
    [rawIndex, m, w].forEach((selected, i) => grids[i].forEach((cell, index) => {
      cell.classList.toggle("active", index === selected);
    }));
    const modelText = m < 0 ? "нет в model50" : `model50[${m}] (${deltaMask[m] ? "дельта" : "абсолют"})`;
    const wireText = `wire51[${w}]${m < 0 ? " (нейтральный слот при model→wire)" : ""}`;
    put("mapping", `${names[rawIndex]}: raw51[${rawIndex}] → ${modelText} → ${wireText}`);
  }
  picker.addEventListener("change", () => selectChannel(Number(picker.value)));
  document.querySelectorAll("[data-channel]").forEach((button) => {
    button.addEventListener("click", () => selectChannel(names.indexOf(button.dataset.channel)));
  });

  selectChannel(names.indexOf("right_wrist_roll"));
  // Reloads may restore input state in some browsers.
  joint = Number(el("joint").value);
  loadMode(document.querySelector('input[name="mode"]:checked').value);
})();
