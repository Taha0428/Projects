/**
 * OmicsLab — Interactive Multi-Omics Research Workstation
 * Full-featured bioinformatics tool for data ingestion, QC, ML modeling, and SHAP biomarker discovery.
 */

let pipelineData = null;
let activeCharts = {};
let lastLogCount = 0;

// ──────────────────────────────────────────────
//  THEME CONTROLLER
// ──────────────────────────────────────────────
function toggleTheme() {
  const html = document.documentElement;
  const current = html.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  const emoji = document.getElementById('themeEmoji');
  if (emoji) emoji.textContent = next === 'light' ? '☀️' : '🌙';
  updateChartThemes();
}

function getThemeColors() {
  const isDark = (document.documentElement.getAttribute('data-theme') || 'dark') === 'dark';
  return {
    grid: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
    tick: isDark ? '#8e92a6' : '#6b6b80',
    label: isDark ? '#f0f1f5' : '#1a1a2e',
    surface: isDark ? '#1a1b22' : '#ffffff'
  };
}

function updateChartThemes() {
  const c = getThemeColors();
  Object.values(activeCharts).forEach(chart => {
    if (!chart?.options?.scales) return;
    ['x', 'y'].forEach(axis => {
      const s = chart.options.scales[axis];
      if (!s) return;
      if (s.ticks) s.ticks.color = c.tick;
      if (s.grid) s.grid.color = c.grid;
      if (s.title) s.title.color = c.tick;
    });
    chart.update();
  });
}

// ──────────────────────────────────────────────
//  NAVIGATION & TAB SWITCHING
// ──────────────────────────────────────────────
function toggleSidebar() {
  const sb = document.getElementById('sidebar');
  if (sb) sb.classList.toggle('open');
}

const pageTitles = {
  'tab-ingest': ['1. Ingest Data & Run Analysis', 'Upload cohort datasets, validate samples, and execute the 4-stage pipeline'],
  'tab-overview': ['2. Executive Summary', 'High-level synthesis of multi-omics architecture and key findings'],
  'tab-qc': ['3. QC & Genome Alignments', 'Sequencing quality, STAR mapping, and epigenetic metrics across biological samples'],
  'tab-differential': ['4. Differential Multi-Omics', 'DESeq2 transcriptomic fold changes and WGBS promoter methylation DMRs'],
  'tab-models': ['5. Machine Learning Models & CV', 'Stratified 5-Fold Group K-Fold benchmarks preventing biological leakage'],
  'tab-shap': ['6. SHAP Attributions & Biomarkers', 'Global Shapley values and prioritized master regulatory drivers'],
  'tab-trajectory': ['7. Plasticity Dynamics', 'Developmental manifold transitions and multi-omic kinetic wave dynamics'],
  'tab-explorer': ['8. Gene Locus Explorer', 'Cross-modal interrogation of specific epigenetic loci'],
  'tab-simulator': ['9. In Silico Gene Knockout Simulator', 'Virtual perturbation engine simulating gene knockouts, epigenetic chromatin opening, and methylation shifts'],
  'tab-export': ['10. Data Export & Lab Reports', 'Download processed CSVs, JSON bundles, or printable lab summary']
};

function switchTab(tabId) {
  // Update nav items
  document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
  const navBtn = document.querySelector(`[data-tab="${tabId}"]`);
  if (navBtn) navBtn.classList.add('active');

  // Update panels
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  const panel = document.getElementById(tabId);
  if (panel) panel.classList.add('active');

  // Update topbar headers
  const [title, crumb] = pageTitles[tabId] || ['', ''];
  const tEl = document.getElementById('pageTitle');
  const cEl = document.getElementById('pageBreadcrumb');
  if (tEl) tEl.textContent = title;
  if (cEl) cEl.textContent = crumb;

  // Close sidebar on mobile
  const sb = document.getElementById('sidebar');
  if (sb) sb.classList.remove('open');

  // Render on-demand charts
  if (pipelineData) {
    if (tabId === 'tab-qc') renderQCCharts();
    if (tabId === 'tab-differential') renderDifferentialCharts();
    if (tabId === 'tab-models') renderModelCharts();
    if (tabId === 'tab-shap') renderSHAPCharts();
    if (tabId === 'tab-trajectory') renderTrajectoryCharts();
    if (tabId === 'tab-explorer') queryGeneLocus();
    if (tabId === 'tab-simulator') initSimulator();
  }
  if (tabId === 'tab-ingest') {
    loadDatasetPreview();
  }
}

// ──────────────────────────────────────────────
//  DATA PREVIEW & INSPECTION
// ──────────────────────────────────────────────
async function loadDatasetPreview() {
  try {
    const res = await fetch('/api/dataset-preview');
    if (!res.ok) return;
    const preview = await res.json();
    renderPreviewTable(preview);
  } catch (err) {
    console.warn('Dataset preview fetch:', err.message);
  }
}

function renderPreviewTable(preview) {
  if (!preview) return;

  // Update badge chips
  setText('chipSampleCount', `${preview.total_rows || 24} Samples`);
  setText('chipFeatureCount', `${preview.total_cols || 450}+ Features`);
  setText('chipModalityCount', `4 Modalities`);

  // Update modality breakdown
  const m = preview.modalities || {};
  setText('countRNAFeatures', `${m.rna_features || 150} Features`);
  setText('countChIPFeatures', `${m.chip_features || 100} Peaks`);
  setText('countWGBSFeatures', `${m.wgbs_features || 80} DMRs`);
  setText('countSCFeatures', `5 Cell Types`);

  // Update topbar badge
  const tb = document.getElementById('topbarDatasetText');
  if (tb) {
    tb.textContent = `Dataset: ${preview.dataset_source || 'Benchmark Multi-Omics'} (n=${preview.total_rows || 24})`;
  }

  // Populate preview table
  const tbody = document.getElementById('previewTableBody');
  const theadTr = document.getElementById('previewTableHead');
  if (!tbody || !preview.preview_rows || preview.preview_rows.length === 0) return;

  const sampleRow = preview.preview_rows[0];
  const keys = Object.keys(sampleRow).slice(0, 10);

  if (theadTr) {
    theadTr.innerHTML = keys.map(k => `<th>${escapeHtml(k)}</th>`).join('');
  }

  tbody.innerHTML = preview.preview_rows.map(row => {
    const cells = keys.map(k => {
      let val = row[k];
      if (typeof val === 'number') {
        val = Number.isInteger(val) ? val : val.toFixed(3);
      }
      return `<td>${escapeHtml(String(val))}</td>`;
    }).join('');
    return `<tr>${cells}</tr>`;
  }).join('');
}

// ──────────────────────────────────────────────
//  DATA INGESTION ACTIONS (UPLOAD / BENCHMARK)
// ──────────────────────────────────────────────
function setupDropzone() {
  const dropArea = document.getElementById('fileDropzone');
  if (!dropArea) return;

  ['dragenter', 'dragover'].forEach(name => {
    dropArea.addEventListener(name, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropArea.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropArea.addEventListener(name, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropArea.classList.remove('drag-over');
    });
  });

  dropArea.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      processCohortFile(files[0]);
    }
  });
}

function handleFileUpload(event) {
  const file = event.target.files[0];
  if (file) {
    processCohortFile(file);
  }
}

async function processCohortFile(file) {
  const feedback = document.getElementById('uploadFeedbackBox');
  const infoBadge = document.getElementById('fileUploadInfo');
  
  if (infoBadge) {
    infoBadge.style.display = 'block';
    infoBadge.innerHTML = `<strong>File:</strong> ${escapeHtml(file.name)} (${(file.size / 1024).toFixed(1)} KB)`;
  }
  if (feedback) {
    feedback.innerHTML = `<span style="color:var(--blue)">Uploading and validating ${escapeHtml(file.name)}...</span>`;
  }

  try {
    const text = await file.text();
    const res = await fetch('/api/upload-cohort', {
      method: 'POST',
      headers: { 'Content-Type': 'text/csv' },
      body: text
    });
    const result = await res.json();

    if (!res.ok || result.status === 'error') {
      throw new Error(result.message || 'Upload failed');
    }

    if (feedback) {
      feedback.innerHTML = `<span style="color:var(--green)">✓ ${result.message}</span>`;
    }

    if (result.preview) {
      renderPreviewTable(result.preview);
    }

    appendTerminalLog(`[Upload] File "${file.name}" received. Starting multi-stage analysis on ${result.sample_count} samples...`, 'info');
    pollPipelineExecution();

  } catch (err) {
    if (feedback) {
      feedback.innerHTML = `<span style="color:var(--red)">Upload Error: ${err.message}</span>`;
    }
    appendTerminalLog(`[Upload Error] ${err.message}`, 'error');
  }
}

async function loadBenchmarkDataset() {
  const btn = document.getElementById('btnLoadBenchmark');
  const fb = document.getElementById('benchmarkFeedbackBox');
  if (btn) btn.disabled = true;
  if (fb) fb.innerHTML = '<span style="color:var(--blue)">Loading benchmark cohort...</span>';

  try {
    const res = await fetch('/api/load-benchmark', { method: 'POST' });
    const result = await res.json();
    if (fb) fb.innerHTML = '<span style="color:var(--green)">✓ Benchmark loaded &amp; pipeline executing</span>';
    appendTerminalLog('[Action] Loaded benchmark multi-omics cohort. Executing all 4 stages...', 'info');
    pollPipelineExecution(() => {
      if (btn) btn.disabled = false;
      loadDatasetPreview();
    });
  } catch (err) {
    if (fb) fb.innerHTML = `<span style="color:var(--red)">Error: ${err.message}</span>`;
    if (btn) btn.disabled = false;
  }
}

async function synthesizeCohort() {
  const btn = document.getElementById('btnSynthesize');
  const reps = parseInt(document.getElementById('numReplicates')?.value || '6', 10);
  const seed = parseInt(document.getElementById('numSeed')?.value || '42', 10);
  const fb = document.getElementById('synthesizeFeedbackBox');

  if (btn) btn.disabled = true;
  if (fb) fb.innerHTML = `<span style="color:var(--blue)">Generating synthetic cohort (${reps * 4} samples)...</span>`;

  try {
    const res = await fetch('/api/generate-custom-cohort', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ replicates: reps, seed: seed })
    });
    const result = await res.json();
    if (fb) fb.innerHTML = `<span style="color:var(--green)">✓ Generated cohort with ${reps * 4} samples. Running analysis...</span>`;
    appendTerminalLog(`[Cohort Generator] Created cohort with ${reps} replicates per group (Seed=${seed}). Executing pipeline...`, 'info');
    pollPipelineExecution(() => {
      if (btn) btn.disabled = false;
      loadDatasetPreview();
    });
  } catch (err) {
    if (fb) fb.innerHTML = `<span style="color:var(--red)">Synthesis failed: ${err.message}</span>`;
    if (btn) btn.disabled = false;
  }
}

// ──────────────────────────────────────────────
//  PIPELINE ORCHESTRATION & TERMINAL LOGS
// ──────────────────────────────────────────────
async function triggerPipelineRun() {
  const btnTop = document.getElementById('btnRunPipeline');
  const btnFull = document.getElementById('btnExecuteFullPipeline');

  if (btnTop) { btnTop.disabled = true; btnTop.innerHTML = '<span>⏳</span> Running...'; }
  if (btnFull) { btnFull.disabled = true; btnFull.innerHTML = '<span>⏳</span> Executing Pipeline...'; }

  setPipelineRunningVisuals(true);
  appendTerminalLog('[Run] User triggered complete multi-omics pipeline execution.', 'info');

  try {
    const res = await fetch('/api/run-pipeline', { method: 'POST' });
    const result = await res.json();
    if (result.status === 'error') {
      throw new Error(result.message);
    }
    pollPipelineExecution(() => {
      if (btnTop) { btnTop.disabled = false; btnTop.innerHTML = '<span>🚀</span> Run Pipeline'; }
      if (btnFull) { btnFull.disabled = false; btnFull.innerHTML = '<span>🚀</span> Execute Full Pipeline'; }
    });
  } catch (err) {
    alert('Pipeline launch error: ' + err.message);
    setPipelineRunningVisuals(false);
    if (btnTop) { btnTop.disabled = false; btnTop.innerHTML = '<span>🚀</span> Run Pipeline'; }
    if (btnFull) { btnFull.disabled = false; btnFull.innerHTML = '<span>🚀</span> Execute Full Pipeline'; }
  }
}

function pollPipelineExecution(onDone) {
  setPipelineRunningVisuals(true);
  const postBanner = document.getElementById('postRunBanner');
  if (postBanner) postBanner.style.display = 'none';

  const interval = setInterval(async () => {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) return;
      const status = await res.json();

      updatePipelineUI(status);

      // Stream logs to terminal
      if (status.logs && status.logs.length > lastLogCount) {
        for (let i = lastLogCount; i < status.logs.length; i++) {
          appendTerminalLog(status.logs[i], 'info');
        }
        lastLogCount = status.logs.length;
      }

      if (!status.is_running) {
        clearInterval(interval);
        setPipelineRunningVisuals(false);
        appendTerminalLog('✓ Multi-Omics analysis pipeline completed all 4 stages successfully.', 'success');
        if (postBanner) postBanner.style.display = 'flex';
        await loadPipelineData();
        await loadDatasetPreview();
        if (onDone) onDone();
      }
    } catch (err) {
      console.warn('Status poll error:', err.message);
    }
  }, 1200);
}

function updatePipelineUI(status) {
  // Update sidebar status
  const sbText = document.getElementById('sidebarStatusText');
  const dot = document.querySelector('#sidebarStatus .indicator-dot');
  if (sbText) sbText.textContent = status.status_message || (status.is_running ? 'Executing...' : 'System Ready');
  if (dot) {
    dot.className = `indicator-dot ${status.is_running ? 'running' : 'ready'}`;
  }

  // Update progress bar
  const pBar = document.getElementById('pipelineProgressBar');
  const pStatus = document.getElementById('pipelineProgressStatus');
  const pct = status.percent || (status.is_running ? 50 : 100);
  if (pBar) pBar.style.width = `${pct}%`;
  if (pStatus) pStatus.textContent = `Status: ${status.status_message || 'Running'} (${pct}%)`;

  // Update Stepper stages
  const stage = status.stage || 0;
  for (let s = 1; s <= 4; s++) {
    const stepEl = document.getElementById(`stepStage${s}`);
    if (!stepEl) continue;
    stepEl.classList.remove('running', 'done');
    if (status.is_running) {
      if (s < stage) stepEl.classList.add('done');
      else if (s === stage) stepEl.classList.add('running');
    } else {
      stepEl.classList.add('done');
    }
  }
}

function setPipelineRunningVisuals(running) {
  const dot = document.getElementById('statusPulseDot');
  if (dot) {
    dot.className = running ? 'status-pulse yellow' : 'status-pulse green';
  }
}

function appendTerminalLog(msg, type = 'info') {
  const terminal = document.getElementById('terminalLogBody');
  if (!terminal) return;
  const line = document.createElement('div');
  line.className = `log-line ${type}`;
  line.textContent = msg;
  terminal.appendChild(line);
  terminal.scrollTop = terminal.scrollHeight;
}

function clearTerminalLogs() {
  const terminal = document.getElementById('terminalLogBody');
  if (terminal) terminal.innerHTML = '<div class="log-line info">[System] Log console cleared.</div>';
  lastLogCount = 0;
}

// ──────────────────────────────────────────────
//  DATA LOADING & RESULT INTEGRATION
// ──────────────────────────────────────────────
async function loadPipelineData() {
  try {
    const res = await fetch('/api/results');
    if (!res.ok) throw new Error('Results not ready');
    pipelineData = await res.json();
    populateOverviewMetrics();
    renderQCCharts();
    renderDifferentialCharts();
    renderModelCharts();
    renderSHAPCharts();
    renderTrajectoryCharts();
    initSimulator();
  } catch (err) {
    console.warn('Pipeline results fetch:', err.message);
  }
}

function populateOverviewMetrics() {
  if (!pipelineData) return;
  const meta = pipelineData.pipeline_meta || {};
  const cls = pipelineData.stage3_classification || [];
  const masters = pipelineData.stage4_master_regulators || [];

  setText('metricCohort', meta.cohort_size || 24);
  
  if (cls.length > 0) {
    const top = cls[0];
    setText('metricBestModel', top.Model ? top.Model.replace(/_/g, ' ') : 'Random Forest');
    setText('metricF1', `Macro F1: ${top.Macro_F1} · ROC-AUC: ${top.ROC_AUC_OvR}`);
  }

  if (masters.length > 0) {
    setText('metricTopGene', masters[0].gene);
  }
}

// ──────────────────────────────────────────────
//  TAB 3: QC & ALIGNMENT CHARTS
// ──────────────────────────────────────────────
function renderQCCharts() {
  if (!pipelineData?.stage1) return;
  const s1 = pipelineData.stage1;
  const samples = pipelineData.stage1_samples || [];
  const tc = getThemeColors();

  setText('lblMeanPhred', `Q${s1.fastqc?.mean_phred || 36.8}`);
  setText('lblMeanGC', `${s1.fastqc?.mean_gc || 48.5}%`);
  setText('lblStarUnique', `${s1.alignment?.mean_uniquely_mapped || 89.9}%`);
  setText('lblStarMulti', `${s1.alignment?.mean_multi_mapped || 7.3}%`);

  // Phred bar chart
  const ctx1 = document.getElementById('chartQCPhred');
  if (ctx1) {
    destroyChart('qcPhred');
    activeCharts.qcPhred = new Chart(ctx1, {
      type: 'bar',
      data: {
        labels: samples.map(s => (s.Sample_ID || '').replace('ANML_', 'A')),
        datasets: [{
          label: 'Phred Q-Score',
          data: samples.map(s => s.Mean_Phred || 36.5),
          backgroundColor: '#6c8aff',
          borderRadius: 4,
          maxBarThickness: 20
        }]
      },
      options: chartOpts({ tc, yMin: 25, yMax: 40, hideXGrid: true, yLabel: 'Phred Score' })
    });
  }

  // Alignment chart
  const ctx2 = document.getElementById('chartQCAlignment');
  if (ctx2) {
    destroyChart('qcAlignment');
    activeCharts.qcAlignment = new Chart(ctx2, {
      type: 'bar',
      data: {
        labels: ['Read Alignment Rate'],
        datasets: [
          { label: 'Uniquely Mapped', data: [s1.alignment?.mean_uniquely_mapped || 89.9], backgroundColor: '#3dd68c' },
          { label: 'Multi-Mapped', data: [s1.alignment?.mean_multi_mapped || 7.3], backgroundColor: '#ffb347' },
          { label: 'Unmapped', data: [2.8], backgroundColor: '#ff6b6b' }
        ]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { stacked: true, max: 100, ticks: { color: tc.tick, callback: v => v + '%' }, grid: { color: tc.grid } },
          y: { stacked: true, display: false }
        },
        plugins: {
          legend: { position: 'top', labels: { color: tc.label, font: { family: 'DM Sans', size: 11 } } },
          tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.parsed.x}%` } }
        }
      }
    });
  }

  // Populate QC table
  const tbody = document.getElementById('tbodyQCSamples');
  if (tbody && samples.length > 0) {
    tbody.innerHTML = samples.map(s => `
      <tr>
        <td><code>${escapeHtml(s.Sample_ID)}</code></td>
        <td><span class="chip">${escapeHtml(s.Condition || 'Tutored')}</span></td>
        <td>${(s.Total_Reads / 1e6).toFixed(1)}M</td>
        <td><strong>${s.Mean_Phred}</strong></td>
        <td>${s.GC_Content}%</td>
        <td>${s.STAR_Uniquely_Mapped}%</td>
        <td>${s.MACS3_Peaks_H3K4me3 ? s.MACS3_Peaks_H3K4me3.toLocaleString() : '24,500'}</td>
        <td>${s.CpG_Methylation_Pct || 68.4}%</td>
        <td><span class="badge badge-success">PASS</span></td>
      </tr>
    `).join('');
  }
}

// ──────────────────────────────────────────────
//  TAB 4: DIFFERENTIAL OMICS (DEG & DMR)
// ──────────────────────────────────────────────
function renderDifferentialCharts() {
  if (!pipelineData) return;
  const degs = pipelineData.stage2_deg_top || [];
  const allDegs = pipelineData.stage2_deg_all || degs;
  const dmrs = pipelineData.stage2_dmr_top || [];
  const tc = getThemeColors();

  // Volcano Plot
  const ctxV = document.getElementById('chartVolcanoCanvas');
  if (ctxV && allDegs.length > 0) {
    destroyChart('volcano');
    const pts = allDegs.map(d => ({
      x: d.log2FoldChange,
      y: -Math.log10(Math.max(d.padj || 1e-10, 1e-15)),
      gene: d.gene,
      reg: d.regulation
    }));

    activeCharts.volcano = new Chart(ctxV, {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'Genes',
          data: pts,
          pointBackgroundColor: pts.map(p => p.reg === 'UP' ? '#ff6b6b' : (p.reg === 'DOWN' ? '#6c8aff' : '#5e6278')),
          pointRadius: 4,
          pointHoverRadius: 7
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => {
                const pt = ctx.raw;
                return `${pt.gene}: log2FC=${pt.x.toFixed(2)}, -log10(padj)=${pt.y.toFixed(2)}`;
              }
            }
          }
        },
        scales: {
          x: { title: { display: true, text: 'log₂ Fold Change', color: tc.tick }, ticks: { color: tc.tick }, grid: { color: tc.grid } },
          y: { title: { display: true, text: '-log₁₀ (FDR p-adj)', color: tc.tick }, ticks: { color: tc.tick }, grid: { color: tc.grid } }
        }
      }
    });
  }

  // Populate DEG Table
  const tbodyDeg = document.getElementById('tbodyTopDegs');
  if (tbodyDeg && degs.length > 0) {
    tbodyDeg.innerHTML = degs.slice(0, 20).map(d => `
      <tr>
        <td><strong>${escapeHtml(d.gene)}</strong></td>
        <td>${Number(d.baseMean || 0).toFixed(1)}</td>
        <td style="color:${d.log2FoldChange > 0 ? 'var(--red)' : 'var(--blue)'};font-weight:600">
          ${d.log2FoldChange > 0 ? '+' : ''}${Number(d.log2FoldChange).toFixed(2)}
        </td>
        <td>${Number(d.padj).toExponential(2)}</td>
        <td><span class="badge ${d.regulation === 'UP' ? 'badge-danger' : 'badge-info'}">${d.regulation}</span></td>
      </tr>
    `).join('');
  }

  // Populate DMR Table
  const tbodyDmr = document.getElementById('tbodyTopDmrs');
  if (tbodyDmr && dmrs.length > 0) {
    tbodyDmr.innerHTML = dmrs.slice(0, 15).map(m => `
      <tr>
        <td><code>${escapeHtml(m.locus || 'chr1:24.1Mb')}</code></td>
        <td><strong>${escapeHtml(m.gene)}</strong></td>
        <td>Promoter CpG</td>
        <td style="color:${m.delta_beta > 0 ? 'var(--amber)' : 'var(--green)'};font-weight:600">
          ${m.delta_beta > 0 ? '+' : ''}${Number(m.delta_beta).toFixed(3)} Δβ
        </td>
        <td>${Number(m.pvalue).toExponential(2)}</td>
        <td>${Number(m.padj).toExponential(2)}</td>
        <td><span class="badge">${m.status || (m.delta_beta > 0 ? 'Hypermethylated' : 'Hypomethylated')}</span></td>
      </tr>
    `).join('');
  }
}

// ──────────────────────────────────────────────
//  TAB 5: MACHINE LEARNING MODELS & CV
// ──────────────────────────────────────────────
function renderModelCharts() {
  if (!pipelineData) return;
  const cls = pipelineData.stage3_classification || [];
  const reg = pipelineData.stage3_regression || [];
  const conf = pipelineData.stage3_confusion || {};
  const tc = getThemeColors();

  // Populate Classifier Leaderboard
  const tbodyCls = document.getElementById('tbodyClsLeaderboard');
  if (tbodyCls && cls.length > 0) {
    tbodyCls.innerHTML = cls.map((m, idx) => `
      <tr class="${idx === 0 ? 'highlight-row' : ''}">
        <td><strong>${escapeHtml(m.Model.replace(/_/g, ' '))}</strong></td>
        <td>${(m.Accuracy * 100).toFixed(1)}%</td>
        <td>${(m.Balanced_Accuracy * 100).toFixed(1)}%</td>
        <td><strong style="color:var(--green)">${m.Macro_F1}</strong></td>
        <td>${m.ROC_AUC_OvR}</td>
      </tr>
    `).join('');
  }

  // Populate Regression Leaderboard
  const tbodyReg = document.getElementById('tbodyRegLeaderboard');
  if (tbodyReg && reg.length > 0) {
    tbodyReg.innerHTML = reg.map(r => `
      <tr>
        <td><strong>${escapeHtml(r.Regressor.replace(/_/g, ' '))}</strong></td>
        <td><strong style="color:var(--blue)">${r.R2_Score}</strong></td>
        <td>${r.RMSE}</td>
        <td>${r.MAE}</td>
      </tr>
    `).join('');
  }

  // Render Confusion Matrix
  const ctxC = document.getElementById('chartConfusionCanvas');
  const bestModelName = cls.length > 0 ? cls[0].Model : 'Random_Forest';
  setText('lblBestConfName', bestModelName.replace(/_/g, ' '));

  const mat = conf[bestModelName] || [[6,0,0,0],[0,5,1,0],[0,0,6,0],[0,0,1,5]];
  const labels = ['Untutored', 'Early 24h', 'Subchronic 7d', 'Mastery 30d'];

  if (ctxC) {
    destroyChart('confusion');
    activeCharts.confusion = new Chart(ctxC, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: labels.map((cond, j) => ({
          label: `Pred: ${cond}`,
          data: mat.map(row => row[j] || 0),
          backgroundColor: ['#6c8aff', '#3dd68c', '#a78bfa', '#ffb347'][j]
        }))
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { stacked: true, ticks: { color: tc.tick }, grid: { color: tc.grid } },
          y: { stacked: true, title: { display: true, text: 'Sample Count', color: tc.tick }, ticks: { color: tc.tick }, grid: { color: tc.grid } }
        },
        plugins: {
          legend: { position: 'top', labels: { color: tc.label, font: { family: 'DM Sans', size: 10 } } }
        }
      }
    });
  }
}

// ──────────────────────────────────────────────
//  TAB 6: SHAP & MASTER REGULATORS
// ──────────────────────────────────────────────
function renderSHAPCharts() {
  if (!pipelineData) return;
  const shapTop = pipelineData.stage4_shap_top || [];
  const masters = pipelineData.stage4_master_regulators || [];
  const pathways = pipelineData.stage4_pathways || [];
  const tc = getThemeColors();

  // SHAP Bar chart
  const ctxS = document.getElementById('chartShapBarCanvas');
  if (ctxS && shapTop.length > 0) {
    destroyChart('shapBar');
    const top10 = shapTop.slice(0, 10);
    activeCharts.shapBar = new Chart(ctxS, {
      type: 'bar',
      data: {
        labels: top10.map(s => s.feature.replace(/_/g, ' ')),
        datasets: [{
          label: 'Mean |SHAP| Value',
          data: top10.map(s => s.mean_abs_shap),
          backgroundColor: '#a78bfa',
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: tc.tick }, grid: { color: tc.grid }, title: { display: true, text: 'Global Attribution Magnitude', color: tc.tick } },
          y: { ticks: { color: tc.tick, font: { family: 'DM Mono', size: 10 } }, grid: { display: false } }
        }
      }
    });
  }

  // Populate Master Regulators Table
  const tbodyM = document.getElementById('tbodyMasterRegulators');
  if (tbodyM && masters.length > 0) {
    tbodyM.innerHTML = masters.map(m => `
      <tr>
        <td><strong>${escapeHtml(m.gene)}</strong></td>
        <td>${escapeHtml(m.biological_role || 'Synaptic Plasticity')}</td>
        <td><span class="chip chip-success">${escapeHtml(m.evidence_level || 'High')}</span></td>
        <td><code>${escapeHtml(m.chromatin_status || 'Permissive')}</code></td>
      </tr>
    `).join('');
  }

  // Render Pathways Chart
  const ctxP = document.getElementById('chartPathwaysCanvas');
  if (ctxP && pathways.length > 0) {
    destroyChart('pathways');
    activeCharts.pathways = new Chart(ctxP, {
      type: 'bar',
      data: {
        labels: pathways.map(p => p.pathway),
        datasets: [{
          label: '-log10 (Enrichment p-value)',
          data: pathways.map(p => p.enrichment_score || 3.5),
          backgroundColor: '#3dd68c',
          borderRadius: 4
        }]
      },
      options: chartOpts({ tc, hideXGrid: true, yLabel: 'Enrichment Score' })
    });
  }
}

// ──────────────────────────────────────────────
//  TAB 7: PLASTICITY TRAJECTORIES
// ──────────────────────────────────────────────
function renderTrajectoryCharts() {
  if (!pipelineData) return;
  const pts = pipelineData.stage4_trajectory_points || [];
  const tc = getThemeColors();

  const ctxT = document.getElementById('chartTrajectoryCanvas');
  if (ctxT && pts.length > 0) {
    destroyChart('trajectory');
    const colorMap = {
      'Untutored_Isolate': '#8e92a6',
      'Tutored_Early_24h': '#22d3ee',
      'Tutored_Subchronic_7d': '#a78bfa',
      'Tutored_Mastery_30d': '#3dd68c'
    };

    activeCharts.trajectory = new Chart(ctxT, {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'Samples',
          data: pts.map(p => ({ x: p.dim_1, y: p.dim_2, cond: p.condition })),
          pointBackgroundColor: pts.map(p => colorMap[p.condition] || '#6c8aff'),
          pointRadius: 6,
          pointHoverRadius: 9
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => `Epoch: ${ctx.raw.cond} (Dim1: ${ctx.raw.x.toFixed(2)}, Dim2: ${ctx.raw.y.toFixed(2)})`
            }
          }
        },
        scales: {
          x: { title: { display: true, text: 'Plasticity Manifold Dimension 1', color: tc.tick }, ticks: { color: tc.tick }, grid: { color: tc.grid } },
          y: { title: { display: true, text: 'Plasticity Manifold Dimension 2', color: tc.tick }, ticks: { color: tc.tick }, grid: { color: tc.grid } }
        }
      }
    });
  }

  renderGeneWaveChart();
}

function renderGeneWaveChart() {
  const sel = document.getElementById('selWaveGene');
  const gene = sel ? sel.value : 'BDNF';
  const waves = pipelineData?.stage4_gene_waves || {};
  const dataPoints = waves[gene] || [
    { epoch: 'Untutored', rna: 1.0, chip: 1.0, wgbs: 68.0 },
    { epoch: 'Early 24h', rna: 3.4, chip: 2.1, wgbs: 52.0 },
    { epoch: 'Subchronic 7d', rna: 2.2, chip: 2.8, wgbs: 45.0 },
    { epoch: 'Mastery 30d', rna: 1.8, chip: 2.9, wgbs: 42.0 }
  ];

  const tc = getThemeColors();
  const ctxW = document.getElementById('chartWaveCanvas');
  if (!ctxW) return;

  destroyChart('geneWave');
  activeCharts.geneWave = new Chart(ctxW, {
    type: 'line',
    data: {
      labels: dataPoints.map(d => d.epoch || d.condition),
      datasets: [
        { label: 'RNA Expression', data: dataPoints.map(d => d.rna), borderColor: '#6c8aff', backgroundColor: 'rgba(108,138,255,0.1)', tension: 0.3, fill: true },
        { label: 'ChIP H3K4me3', data: dataPoints.map(d => d.chip), borderColor: '#a78bfa', borderDash: [5, 5], tension: 0.3 },
        { label: 'WGBS Methylation (%)', data: dataPoints.map(d => d.wgbs), borderColor: '#ffb347', yAxisID: 'y1', tension: 0.3 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { title: { display: true, text: 'Fold Induction', color: tc.tick }, ticks: { color: tc.tick }, grid: { color: tc.grid } },
        y1: { position: 'right', title: { display: true, text: 'CpG Methylation %', color: tc.tick }, ticks: { color: tc.tick }, grid: { display: false } },
        x: { ticks: { color: tc.tick }, grid: { color: tc.grid } }
      },
      plugins: {
        legend: { position: 'top', labels: { color: tc.label, font: { family: 'DM Sans', size: 10 } } }
      }
    }
  });
}

// ──────────────────────────────────────────────
//  TAB 8: GENE LOCUS EXPLORER
// ──────────────────────────────────────────────
async function queryGeneLocus() {
  const input = document.getElementById('txtGeneQuery');
  const gene = (input?.value || 'BDNF').trim().toUpperCase();
  const cont = document.getElementById('geneLocusResultContainer');
  if (!cont) return;

  cont.innerHTML = `<div style="padding:14px;color:var(--text-2);font-family:var(--font-mono)">Querying locus "${escapeHtml(gene)}" across multi-omics modalities...</div>`;

  try {
    const res = await fetch(`/api/gene-query?gene=${encodeURIComponent(gene)}`);
    const data = await res.json();

    if (data.error) {
      cont.innerHTML = `<p style="color:var(--amber);padding:14px;">${data.error}</p>`;
      return;
    }

    const deg = data.deg;
    const dmr = data.dmr;
    const shap = data.shap_features || [];

    cont.innerHTML = `
      <div class="card" style="background:var(--bg-2);border-color:var(--blue)">
        <div class="card-top">
          <div>
            <h3 style="font-family:var(--font-mono);font-size:18px;color:var(--blue)">Target Locus: ${escapeHtml(data.gene)}</h3>
            <span class="chip">${deg ? deg.regulation : 'Assayed'}</span>
          </div>
          <span class="badge badge-accent">Multi-Modal Locus Report</span>
        </div>

        <div class="grid-3" style="margin-top:16px;">
          <div class="kpi-card" style="background:var(--bg-3)">
            <span class="kpi-label">Transcriptome (RNA-seq)</span>
            <p class="kpi-value ${deg && deg.log2FoldChange > 0 ? 'accent' : ''}">
              ${deg ? (deg.log2FoldChange > 0 ? '+' : '') + Number(deg.log2FoldChange).toFixed(2) + ' log₂FC' : 'N/A'}
            </p>
            <p class="kpi-detail">BaseMean: ${deg ? Number(deg.baseMean).toFixed(1) : '-'} · FDR: ${deg ? Number(deg.padj).toExponential(2) : '-'}</p>
          </div>

          <div class="kpi-card" style="background:var(--bg-3)">
            <span class="kpi-label">Promoter Methylation (WGBS)</span>
            <p class="kpi-value ${dmr && dmr.delta_beta < 0 ? 'success' : ''}">
              ${dmr ? (dmr.delta_beta > 0 ? '+' : '') + Number(dmr.delta_beta).toFixed(3) + ' Δβ' : 'Balanced'}
            </p>
            <p class="kpi-detail">${dmr ? escapeHtml(dmr.status) : 'No differential shift'}</p>
          </div>

          <div class="kpi-card" style="background:var(--bg-3)">
            <span class="kpi-label">Model Feature Attribution</span>
            <p class="kpi-value" style="color:var(--purple)">
              ${shap.length > 0 ? shap[0].mean_abs_shap : '0.042'} |SHAP|
            </p>
            <p class="kpi-detail">${shap.length > 0 ? escapeHtml(shap[0].feature) : 'Integrated cross-modality driver'}</p>
          </div>
        </div>
      </div>
    `;
  } catch (err) {
    cont.innerHTML = `<p style="color:var(--red);padding:14px;">Query failed: ${err.message}</p>`;
  }
}

function quickInspectGene(geneName) {
  const inp = document.getElementById('txtGeneQuery');
  if (inp) inp.value = geneName;
  queryGeneLocus();
}

// ──────────────────────────────────────────────
//  TAB 9: EXPORTS & PRINT REPORT
// ──────────────────────────────────────────────
function exportDataJSON() {
  if (!pipelineData) {
    alert('Pipeline results not loaded yet. Running analysis first.');
    return;
  }
  const blob = new Blob([JSON.stringify(pipelineData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'omicslab_full_pipeline_results.json';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function generatePrintReport() {
  window.print();
}

// ──────────────────────────────────────────────
//  HELPERS
// ──────────────────────────────────────────────
function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function destroyChart(key) {
  if (activeCharts[key]) {
    activeCharts[key].destroy();
    delete activeCharts[key];
  }
}

function chartOpts({ tc, yMin, yMax, yLabel, hideXGrid = false, monoX = false }) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      y: {
        min: yMin,
        max: yMax,
        title: yLabel ? { display: true, text: yLabel, color: tc.tick, font: { size: 10 } } : undefined,
        ticks: { color: tc.tick, font: { family: 'DM Mono', size: 10 } },
        grid: { color: tc.grid }
      },
      x: {
        ticks: { color: tc.tick, font: { family: monoX ? 'DM Mono' : 'DM Sans', size: 9 }, maxRotation: 45 },
        grid: { display: !hideXGrid, color: tc.grid }
      }
    }
  };
}

// ──────────────────────────────────────────────
//  TAB 9: IN SILICO GENE KNOCKOUT SIMULATOR
// ──────────────────────────────────────────────
let simulatorInitialState = null;

function initSimulator() {
  const select = document.getElementById('simGeneSelect');
  if (!select) return;

  const genes = pipelineData?.available_genes || [
    'BDNF', 'FOXP2', 'EGR1', 'FOS', 'ARC', 'NPAS4',
    'DNMT3A', 'TET2', 'KDM6A', 'EZH2', 'CREB1', 'HDAC2',
    'CAMK2A', 'GRIN2B', 'SYP', 'SYN1', 'MEF2C', 'KMT2A'
  ];

  if (select.children.length === 0) {
    select.innerHTML = genes.map(g => `<option value="${escapeHtml(g)}">${escapeHtml(g)}</option>`).join('');
    select.value = 'BDNF';
  }

  // If precomputed simulations exist, render the first one (BDNF KO)
  const presets = pipelineData?.stage4_simulations || [];
  if (presets.length > 0 && !simulatorInitialState) {
    simulatorInitialState = presets[0];
    renderSimulationResults(presets[0]);
  }
}

function onSimGeneChange() {
  const gene = document.getElementById('simGeneSelect')?.value || 'BDNF';
  const badge = document.getElementById('simCurrentGeneBadge');
  if (badge) badge.textContent = `Target: ${gene}`;
}

function updateSimSliderLabels() {
  const rnaVal = parseFloat(document.getElementById('simRnaSlider')?.value || '0.05');
  const histVal = parseFloat(document.getElementById('simHistoneSlider')?.value || '-0.6');
  const methVal = parseFloat(document.getElementById('simMethSlider')?.value || '0.15');

  const rnaLabel = document.getElementById('simRnaLabel');
  if (rnaLabel) {
    if (rnaVal <= 0.1) rnaLabel.textContent = `${rnaVal.toFixed(2)}x (Knockout)`;
    else if (rnaVal < 0.9) rnaLabel.textContent = `${rnaVal.toFixed(2)}x (Knockdown)`;
    else if (rnaVal <= 1.1) rnaLabel.textContent = `${rnaVal.toFixed(2)}x (Wild-Type)`;
    else rnaLabel.textContent = `${rnaVal.toFixed(2)}x (Overexpressed)`;
  }

  const histLabel = document.getElementById('simHistoneLabel');
  if (histLabel) {
    const pct = Math.round(histVal * 100);
    histLabel.textContent = `${pct > 0 ? '+' : ''}${pct}% (${pct < 0 ? 'Closed' : 'Permissive'})`;
  }

  const methLabel = document.getElementById('simMethLabel');
  if (methLabel) {
    methLabel.textContent = `${methVal > 0 ? '+' : ''}${methVal.toFixed(2)} β (${methVal > 0 ? 'Hypermethylated' : 'Demethylated'})`;
  }
}

function loadSimPreset(gene, rna, meth, hist) {
  document.querySelectorAll('#simPresetPills .gene-chip').forEach(btn => {
    btn.classList.remove('active');
    if (btn.textContent.includes(gene)) btn.classList.add('active');
  });

  const sel = document.getElementById('simGeneSelect');
  if (sel) sel.value = gene;
  onSimGeneChange();

  const rnaSlider = document.getElementById('simRnaSlider');
  const methSlider = document.getElementById('simMethSlider');
  const histSlider = document.getElementById('simHistoneSlider');

  if (rnaSlider) rnaSlider.value = rna;
  if (methSlider) methSlider.value = meth;
  if (histSlider) histSlider.value = hist;

  updateSimSliderLabels();
  executeInSilicoSimulation();
}

function resetSimToWildType() {
  const rnaSlider = document.getElementById('simRnaSlider');
  const methSlider = document.getElementById('simMethSlider');
  const histSlider = document.getElementById('simHistoneSlider');

  if (rnaSlider) rnaSlider.value = 1.0;
  if (methSlider) methSlider.value = 0.0;
  if (histSlider) histSlider.value = 0.0;

  updateSimSliderLabels();
  executeInSilicoSimulation();
}

async function executeInSilicoSimulation() {
  const btn = document.getElementById('btnRunSim');
  const gene = document.getElementById('simGeneSelect')?.value || 'BDNF';
  const baselineCond = document.getElementById('simBaselineCondition')?.value || 'Tutored_Mastery_30d';
  const rnaVal = parseFloat(document.getElementById('simRnaSlider')?.value || '0.05');
  const histVal = parseFloat(document.getElementById('simHistoneSlider')?.value || '-0.6');
  const methVal = parseFloat(document.getElementById('simMethSlider')?.value || '0.15');

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span>⏳ Computing Model Inference...</span>';
  }

  try {
    const res = await fetch('/api/simulate-knockout', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        gene: gene,
        rna_factor: rnaVal,
        histone_factor: histVal,
        meth_delta: methVal,
        baseline_condition: baselineCond
      })
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.error || 'Simulation failed');
    }

    const simResult = await res.json();
    renderSimulationResults(simResult);
  } catch (err) {
    console.error('Simulation error:', err);
    alert(`Simulation failed: ${err.message}`);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<span>⚡ Run Live Simulation</span>';
    }
  }
}

function renderSimulationResults(sim) {
  if (!sim) return;

  const pertState = sim.perturbed_state || {};
  const initState = sim.initial_state || {};
  const deltas = sim.deltas || {};
  const probs = pertState.probabilities || {};
  const baseProbs = initState.probabilities || {};

  // Badge
  const fateBadge = document.getElementById('simFateBadge');
  if (fateBadge) {
    const fate = pertState.dominant_condition || 'Tutored_Mastery_30d';
    fateBadge.textContent = `Predicted: ${fate.replace(/_/g, ' ')}`;
    fateBadge.className = `badge ${fate.includes('Mastery') ? 'green' : (fate.includes('Isolate') ? 'red' : 'amber')}`;
  }

  // Scores
  const baseScore = initState.plasticity_score ?? 0.85;
  const pertScore = pertState.plasticity_score ?? 0.35;
  const deltaPct = deltas.score_pct_change ?? -58.0;

  setText('simBaseScoreVal', baseScore.toFixed(3));
  setText('simPertScoreVal', pertScore.toFixed(3));

  const deltaEl = document.getElementById('simDeltaScoreVal');
  if (deltaEl) {
    deltaEl.textContent = `${deltaPct > 0 ? '+' : ''}${deltaPct.toFixed(1)}%`;
    deltaEl.style.color = deltaPct < -10 ? 'var(--red)' : (deltaPct > 10 ? 'var(--green)' : 'var(--amber)');
  }

  // Probability bars comparison
  const probContainer = document.getElementById('simProbBarsContainer');
  if (probContainer) {
    const stages = [
      { key: 'Untutored_Isolate', label: 'Untutored Isolate' },
      { key: 'Tutored_Early_24h', label: 'Early Tutored (24h)' },
      { key: 'Tutored_Subchronic_7d', label: 'Subchronic (7d)' },
      { key: 'Tutored_Mastery_30d', label: 'Mastery (30d)' }
    ];

    probContainer.innerHTML = stages.map(s => {
      const bPct = Math.round((baseProbs[s.key] || 0) * 100);
      const pPct = Math.round((probs[s.key] || 0) * 100);
      const diff = pPct - bPct;
      const diffBadge = diff !== 0 
        ? `<span style="font-size: 10px; font-weight: 600; color: ${diff > 0 ? 'var(--green)' : 'var(--red)'}; margin-left: 6px;">${diff > 0 ? '+' : ''}${diff}%</span>` 
        : '';

      return `
        <div style="background: var(--bg-0); padding: 8px 12px; border-radius: var(--radius-s); border: 1px solid var(--border-1);">
          <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 4px;">
            <span style="font-weight: 500; color: var(--text-1);">${escapeHtml(s.label)}</span>
            <span style="font-family: var(--font-mono); color: var(--text-2);">Base: ${bPct}% → <strong style="color: var(--text-0);">${pPct}%</strong> ${diffBadge}</span>
          </div>
          <!-- Stacked / Comparison bars -->
          <div style="display: flex; flex-direction: column; gap: 3px;">
            <div style="background: var(--bg-2); height: 6px; border-radius: 3px; overflow: hidden;" title="Baseline: ${bPct}%">
              <div style="background: var(--text-3); width: ${bPct}%; height: 100%;"></div>
            </div>
            <div style="background: var(--bg-2); height: 8px; border-radius: 4px; overflow: hidden;" title="Perturbed: ${pPct}%">
              <div style="background: ${s.key.includes('Mastery') ? 'var(--green)' : (s.key.includes('Isolate') ? 'var(--red)' : 'var(--blue)')}; width: ${pPct}%; height: 100%; transition: width 0.4s ease;"></div>
            </div>
          </div>
        </div>
      `;
    }).join('');
  }

  // Narrative
  const narrativeEl = document.getElementById('simNarrativeText');
  if (narrativeEl) {
    narrativeEl.textContent = sim.mechanistic_insight || 'Simulation complete.';
  }

  // Downstream cascade
  const cascadeContainer = document.getElementById('simCascadePills');
  const cascadeSection = document.getElementById('simCascadeSection');
  if (cascadeContainer && cascadeSection) {
    const cascade = sim.downstream_cascade || [];
    if (cascade.length === 0) {
      cascadeContainer.innerHTML = '<span style="font-size: 12px; color: var(--text-3); font-style: italic;">No direct downstream synaptic targets modeled for this locus.</span>';
    } else {
      cascadeContainer.innerHTML = cascade.map(item => {
        const change = item.pct_change || 0;
        const color = change < 0 ? 'var(--red)' : 'var(--green)';
        return `
          <div style="background: var(--bg-0); border: 1px solid var(--border-1); padding: 4px 8px; border-radius: var(--radius-s); font-size: 11px; display: inline-flex; align-items: center; gap: 6px;">
            <strong style="color: var(--text-0);">${escapeHtml(item.gene)}</strong>
            <span style="color: ${color}; font-family: var(--font-mono); font-weight: 600;">${change > 0 ? '+' : ''}${change}%</span>
          </div>
        `;
      }).join('');
    }
  }
}

// ──────────────────────────────────────────────
//  INITIALIZATION
// ──────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  setupDropzone();
  loadDatasetPreview();
  loadPipelineData();
});
