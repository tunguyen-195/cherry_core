const appState = {
  currentJobId: null,
  inventory: null,
  pollHandle: null,
  lastResult: null,
};

const MODEL_LABELS = {
  phowhisper: "PhoWhisper",
  "whisper-v2": "Whisper V2",
  whisperx: "WhisperX",
  "stable-ts": "Ổn định mốc thời gian",
  speechbrain: "SpeechBrain",
  "silero-vad": "Lọc khoảng lặng",
  protonx: "Giảm ảo giác ngữ cảnh",
  vistral: "Mô-đun phân tích cục bộ",
};

const FAMILY_LABELS = {
  asr: "ASR",
  refinement: "Tinh chỉnh",
  diarization: "Người nói",
  vad: "Tiền xử lý",
  correction: "Hiệu chỉnh",
  llm: "Phân tích",
};

document.addEventListener("DOMContentLoaded", () => {
  bindTabs();
  bindForm();
  bindStepButtons();
  bindScenarioHint();
  loadModelInventory();
  renderIntelSummary(null);
});

async function loadModelInventory() {
  try {
    const response = await fetch("/api/models");
    const payload = await response.json();
    appState.inventory = payload;
    renderModelStatus(payload.items || []);
    applyCapabilityLocks(payload.capabilities || {});
    updateStepButtons(appState.lastResult || {});
  } catch (error) {
    setError(`Không tải được inventory model: ${error.message}`);
  }
}

function renderModelStatus(items) {
  const container = document.getElementById("modelStatus");
  container.innerHTML = "";

  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = `status-card ${item.offline_ready ? "ready" : "missing"}`;
    const displayName = MODEL_LABELS[item.model_id] || item.model_id;
    card.innerHTML = `
      <div class="status-head">
        <strong>${displayName}</strong>
        <span class="status-pill ${item.offline_ready ? "status-on" : "status-off"}">${item.offline_ready ? "Sẵn sàng" : "Thiếu"}</span>
      </div>
      <div class="status-family">${FAMILY_LABELS[item.family] || item.family}</div>
      <small>${item.notes || ""}</small>
    `;
    container.appendChild(card);
  });
}

function applyCapabilityLocks(capabilities) {
  toggleOption("asrEngine", "phowhisper", capabilities.asr_engines?.phowhisper);
  toggleOption("asrEngine", "whisper-v2", capabilities.asr_engines?.["whisper-v2"]);
  toggleOption("asrEngine", "whisperx", capabilities.asr_engines?.whisperx);
  toggleOption("device", "cuda", capabilities.devices?.cuda);
  toggleOption("speakerMode", "speechbrain", capabilities.speaker_modes?.speechbrain);
  applyPreferredDevice(capabilities.devices || {});

  toggleCheckbox("applyVad", capabilities.features?.apply_vad);
  toggleCheckbox("applyProtonx", capabilities.features?.apply_protonx);
  toggleCheckbox("applyLlmCorrection", capabilities.features?.apply_llm_correction);
  toggleCheckbox("speakerRefine", capabilities.features?.speaker_refine);
}

function applyPreferredDevice(devices) {
  const select = document.getElementById("device");
  if (!select) {
    return;
  }

  select.value = devices.cuda ? "cuda" : "cpu";
}

function toggleOption(selectId, optionValue, enabled) {
  const select = document.getElementById(selectId);
  const option = [...select.options].find((item) => item.value === optionValue);
  if (!option) {
    return;
  }

  option.disabled = !enabled;
  if (!enabled && select.value === optionValue) {
    const fallback = [...select.options].find((item) => !item.disabled);
    if (fallback) {
      select.value = fallback.value;
    }
  }
}

function toggleCheckbox(id, enabled) {
  const input = document.getElementById(id);
  input.disabled = !enabled;
  if (!enabled) {
    input.checked = false;
  }
}

function bindScenarioHint() {
  const select = document.getElementById("analysisScenario");
  const hint = document.getElementById("scenarioHint");

  const update = () => {
    const label = select.options[select.selectedIndex]?.textContent || "kịch bản đã chọn";
    hint.textContent = `Bản tóm tắt trinh sát sẽ ưu tiên logic suy diễn theo: ${label}.`;
  };

  select.addEventListener("change", update);
  update();
}

function bindForm() {
  const form = document.getElementById("jobForm");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearError();

    if (!appState.inventory) {
      await loadModelInventory();
    }

    const fileInput = document.getElementById("audioFile");
    if (!fileInput.files.length) {
      setError("Cần chọn file audio trước khi chạy.");
      return;
    }

    const asrEngine = document.getElementById("asrEngine");
    const selectedAsr = [...asrEngine.options].find((item) => item.value === asrEngine.value);
    if (selectedAsr?.disabled) {
      const fallbackAsr = [...asrEngine.options].find((item) => !item.disabled);
      if (!fallbackAsr) {
        setError("Không có ASR engine offline nào đang sẵn sàng.");
        return;
      }
      asrEngine.value = fallbackAsr.value;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("asr_engine", asrEngine.value);
    formData.append("analysis_scenario", document.getElementById("analysisScenario").value);
    formData.append("apply_vad", checkboxValue("applyVad"));
    formData.append("apply_hallucination_filter", checkboxValue("applyHallucinationFilter"));
    formData.append("apply_domain_postprocess", checkboxValue("applyDomainPostprocess"));
    formData.append("domain", document.getElementById("domain").value);
    formData.append("apply_protonx", checkboxValue("applyProtonx"));
    formData.append("apply_llm_correction", checkboxValue("applyLlmCorrection"));
    formData.append("speaker_mode", document.getElementById("speakerMode").value);
    formData.append("speaker_refine", checkboxValue("speakerRefine"));
    formData.append("device", document.getElementById("device").value);

    disableAllActions(true);

    try {
      const response = await fetch("/api/jobs", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Tạo job thất bại.");
      }

      appState.currentJobId = payload.job_id;
      appState.lastResult = null;
      renderResult({});
      updateJobStatus(payload);
      startPolling(payload.job_id);
    } catch (error) {
      disableAllActions(false);
      setError(error.message);
    }
  });
}

function bindStepButtons() {
  document.querySelectorAll("[data-step]").forEach((button) => {
    button.addEventListener("click", async () => {
      if (!appState.currentJobId) {
        return;
      }

      clearError();
      disableAllActions(true);
      const step = button.dataset.step;

      try {
        const response = await fetch(`/api/jobs/${appState.currentJobId}/steps/${step}`, {
          method: "POST",
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "Chạy step thất bại.");
        }
        updateJobStatus(payload);
        startPolling(appState.currentJobId);
      } catch (error) {
        disableAllActions(false);
        setError(error.message);
      }
    });
  });
}

function bindTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
      button.classList.add("active");
      document.getElementById(`tab-${button.dataset.tab}`).classList.add("active");
    });
  });
}

function startPolling(jobId) {
  stopPolling();
  appState.pollHandle = window.setInterval(async () => {
    try {
      const response = await fetch(`/api/jobs/${jobId}`);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Không đọc được trạng thái job.");
      }
      updateJobStatus(payload);
      if (payload.state === "completed" || payload.state === "failed") {
        stopPolling();
        disableAllActions(false);
        await loadResult(jobId);
      }
    } catch (error) {
      stopPolling();
      disableAllActions(false);
      setError(error.message);
    }
  }, 1500);
}

function stopPolling() {
  if (appState.pollHandle) {
    window.clearInterval(appState.pollHandle);
    appState.pollHandle = null;
  }
}

async function loadResult(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}/result`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Không tải được kết quả.");
    }
    appState.lastResult = payload;
    renderResult(payload);
    updateStepButtons(payload);
  } catch (error) {
    setError(error.message);
  }
}

function renderResult(result) {
  document.getElementById("rawText").textContent = result.raw_text || "";
  document.getElementById("stableText").textContent = result.stable_text || "";
  document.getElementById("filteredText").textContent = result.filtered_text || "";
  document.getElementById("correctedText").textContent = result.corrected_text || "";
  document.getElementById("speakerText").textContent = result.speaker_transcript || "";
  renderIntelSummary(result);

  const downloadContainer = document.getElementById("downloadLinks");
  downloadContainer.innerHTML = "";
  Object.entries(result.downloads || {}).forEach(([label, url]) => {
    const link = document.createElement("a");
    link.href = url;
    link.textContent = label;
    link.target = "_blank";
    downloadContainer.appendChild(link);
  });
}

function renderIntelSummary(result) {
  const empty = document.getElementById("intelEmpty");
  const panel = document.getElementById("intelPanel");
  const report = result?.intel_report || {};
  const strategic = report.strategic_assessment || {};
  const conclusion = strategic.final_conclusion || {};
  const recommendations = report.operational_recommendations || [];
  const cards = result?.intel_cards || [];
  const timeline = result?.intel_timeline || [];
  const riskFlags = result?.risk_flags || [];

  document.getElementById("intelSummaryText").textContent = result?.intel_summary || "";

  if (!result?.intel_summary) {
    empty.classList.remove("hidden");
    panel.classList.add("hidden");
    document.getElementById("intelBriefing").textContent = "";
    document.getElementById("intelConclusion").textContent = "";
    document.getElementById("intelRecommendations").innerHTML = "";
    document.getElementById("intelMetrics").innerHTML = "";
    document.getElementById("intelRiskFlags").innerHTML = "";
    document.getElementById("intelCardGrid").innerHTML = "";
    document.getElementById("intelTimeline").innerHTML = "";
    return;
  }

  empty.classList.add("hidden");
  panel.classList.remove("hidden");

  document.getElementById("intelScenarioBadge").textContent = `Kịch bản: ${formatScenario(result.metadata?.analysis_scenario)}`;
  document.getElementById("intelThreatBadge").textContent = `Threat: ${strategic.threat_level || "UNKNOWN"}`;
  document.getElementById("intelClassBadge").textContent = strategic.classification || "Chưa phân loại";
  document.getElementById("intelBriefing").textContent = strategic.executive_briefing || result.intel_summary;
  document.getElementById("intelConclusion").textContent = [
    conclusion.verdict || "Không có kết luận",
    conclusion.investigator_note || "",
  ].filter(Boolean).join(" | ");

  renderIntelMetrics(cards, timeline, riskFlags);
  renderRiskFlags(riskFlags);
  renderIntelCards(cards);
  renderIntelTimeline(timeline);
  renderRecommendations(recommendations);
}

function renderIntelMetrics(cards, timeline, riskFlags) {
  const container = document.getElementById("intelMetrics");
  container.innerHTML = "";

  const totalItems = cards.reduce((count, card) => count + (card.items || []).length, 0);
  [
    { label: "Mục trích xuất", value: totalItems },
    { label: "Timeline", value: timeline.length },
    { label: "Cảnh báo", value: riskFlags.length },
  ].forEach((metric) => {
    const chip = document.createElement("span");
    chip.className = "metric-chip";
    chip.textContent = `${metric.label}: ${metric.value}`;
    container.appendChild(chip);
  });
}

function renderRiskFlags(flags) {
  const container = document.getElementById("intelRiskFlags");
  container.innerHTML = "";

  if (!flags.length) {
    const pill = document.createElement("span");
    pill.className = "risk-flag low";
    pill.textContent = "Chưa có cảnh báo nổi bật";
    container.appendChild(pill);
    return;
  }

  flags.forEach((flag) => {
    const pill = document.createElement("span");
    pill.className = `risk-flag ${flag.level || "medium"}`;
    pill.textContent = flag.detail ? `${flag.label}: ${flag.detail}` : flag.label;
    container.appendChild(pill);
  });
}

function renderIntelCards(cards) {
  const container = document.getElementById("intelCardGrid");
  container.innerHTML = "";

  if (!cards.length) {
    container.appendChild(renderEmptyIntelCard("Không có dữ liệu intelligence cấu trúc."));
    return;
  }

  cards.forEach((card) => {
    const article = document.createElement("article");
    article.className = "intel-card";

    const title = document.createElement("h3");
    title.textContent = card.title || "Thông tin trích xuất";
    article.appendChild(title);

    if (card.summary) {
      const summary = document.createElement("p");
      summary.className = "hint";
      summary.textContent = card.summary;
      article.appendChild(summary);
    }

    const list = document.createElement("div");
    list.className = "intel-card-list";

    (card.items || []).forEach((item) => {
      const row = document.createElement("article");
      row.className = "intel-card-item";

      const label = document.createElement("strong");
      label.textContent = item.label || "Không rõ nhãn";
      row.appendChild(label);

      const value = document.createElement("span");
      value.textContent = item.value || "Không có thông tin";
      row.appendChild(value);

      if (item.meta) {
        const meta = document.createElement("small");
        meta.textContent = item.meta;
        row.appendChild(meta);
      }

      list.appendChild(row);
    });

    article.appendChild(list);
    container.appendChild(article);
  });
}

function renderIntelTimeline(items) {
  const container = document.getElementById("intelTimeline");
  container.innerHTML = "";

  if (!items.length) {
    container.appendChild(renderEmptyIntelCard("Chưa dựng được timeline rõ ràng từ transcript hiện tại."));
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("article");
    row.className = "timeline-item";

    const time = document.createElement("div");
    time.className = "timeline-time";
    time.textContent = item.time || "Chưa rõ thời gian";
    row.appendChild(time);

    const body = document.createElement("div");
    body.className = "timeline-body";

    const title = document.createElement("strong");
    title.textContent = item.title || "Diễn biến được nhắc tới";
    body.appendChild(title);

    const metaParts = [];
    if (item.location) {
      metaParts.push(`Địa điểm: ${item.location}`);
    }
    if ((item.actors || []).length) {
      metaParts.push(`Liên quan: ${item.actors.join(", ")}`);
    }
    if (item.detail) {
      metaParts.push(item.detail);
    }

    const meta = document.createElement("div");
    meta.className = "timeline-meta";
    meta.textContent = metaParts.join(" | ") || "Không có chi tiết bổ sung.";
    body.appendChild(meta);

    row.appendChild(body);
    container.appendChild(row);
  });
}

function renderRecommendations(recommendations) {
  const list = document.getElementById("intelRecommendations");
  list.innerHTML = "";

  if (!recommendations.length) {
    const item = document.createElement("li");
    item.textContent = "Chưa có khuyến nghị tác nghiệp.";
    list.appendChild(item);
    return;
  }

  recommendations.forEach((text) => {
    const item = document.createElement("li");
    item.textContent = text;
    list.appendChild(item);
  });
}

function renderEmptyIntelCard(message) {
  const article = document.createElement("article");
  article.className = "intel-card intel-card-full";
  const text = document.createElement("p");
  text.textContent = message;
  article.appendChild(text);
  return article;
}

function updateJobStatus(job) {
  document.getElementById("jobId").textContent = job.job_id || "Chưa có";
  document.getElementById("jobState").textContent = job.state || "idle";
  document.getElementById("jobStage").textContent = formatStage(job.stage);
  document.getElementById("progressBar").style.width = `${job.progress || 0}%`;
  document.getElementById("jobError").textContent = job.error || "";
}

function updateStepButtons(result) {
  const caps = appState.inventory?.capabilities || {};
  const running = document.getElementById("jobState").textContent.toLowerCase() === "running";
  const hasJob = Boolean(appState.currentJobId);
  const hasTranscript = Boolean(result.raw_text || result.filtered_text);

  document.querySelectorAll("[data-step]").forEach((button) => {
    const step = button.dataset.step;
    let enabled = hasJob && !running && hasTranscript;

    if (step === "stable_ts") {
      enabled = enabled && !!caps.features?.apply_stable_ts && result.metadata?.asr_engine === "whisper-v2";
    }
    if (step === "protonx") {
      enabled = enabled && !!caps.features?.apply_protonx;
    }
    if (step === "llm_correction") {
      enabled = enabled && !!caps.features?.apply_llm_correction;
    }
    if (step === "intel_summary") {
      enabled = enabled && !!caps.features?.apply_intel_summary;
    }
    if (step === "diarization") {
      enabled = enabled && !!caps.speaker_modes?.speechbrain;
    }
    if (step === "speaker_refine") {
      enabled = enabled && !!caps.features?.speaker_refine && (result.speaker_segments || []).length > 0;
    }

    button.disabled = !enabled;
  });
}

function disableAllActions(disabled) {
  document.getElementById("submitButton").disabled = disabled;
  document.querySelectorAll("[data-step]").forEach((button) => {
    button.disabled = disabled;
  });
}

function checkboxValue(id) {
  return document.getElementById(id).checked ? "true" : "false";
}

function setError(message) {
  document.getElementById("jobError").textContent = message;
}

function clearError() {
  document.getElementById("jobError").textContent = "";
}

function formatStage(stage) {
  const labels = {
    queued: "Đang chờ",
    starting: "Khởi động",
    normalize: "Chuẩn hóa audio",
    vad: "Lọc khoảng lặng",
    transcribe: "Phiên âm",
    filter: "Lọc nhiễu cơ bản",
    stable_ts: "Ổn định mốc thời gian",
    protonx: "Giảm ảo giác ngữ cảnh",
    llm_correction: "Chuẩn hóa ngữ nghĩa AI",
    intel_summary: "Tóm tắt trinh sát",
    diarization: "Tách người nói",
    speaker_refine: "Suy luận vai trò người nói",
    export: "Xuất kết quả",
    completed: "Hoàn tất",
    failed: "Thất bại",
  };
  return labels[stage] || stage || "-";
}

function formatScenario(value) {
  const labels = {
    general_intelligence: "Trinh sát tổng hợp",
    drug_trafficking: "Ma túy/đường dây",
    high_tech_fraud: "Gian lận công nghệ",
  };
  return labels[value] || value || "Không xác định";
}
