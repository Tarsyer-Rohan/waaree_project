// ==================== Populate Filters ====================

async function populateLines() {
  try {
    console.log('Fetching production lines from:', "/api/production_lines");
    const res = await fetch("/api/production_lines");
    const json = await res.json();
    console.log('Production lines response:', json);
    if (!json.ok) return;

    const select = document.getElementById("line");
    select.innerHTML = '<option>All Lines</option>';

    json.lines.forEach((line) => {
      const opt = document.createElement("option");
      opt.value = line;
      opt.text = `Line ${line}`;
      select.appendChild(opt);
    });
    console.log('Populated lines dropdown with:', json.lines);
  } catch (err) {
    console.error("Error fetching lines:", err);
  }
}

async function populateZones() {
  try {
    console.log('Fetching production zones from:', "/api/production_zones");
    const res = await fetch("/api/production_zones");
    const json = await res.json();
    console.log('Production zones response:', json);
    if (!json.ok) return;

    const select = document.getElementById("zone");
    select.innerHTML = '<option>All Zones</option>';

    json.zones.forEach((zone) => {
      const opt = document.createElement("option");
      opt.value = zone;
      opt.text = zone;
      select.appendChild(opt);
    });
    console.log('Populated zones dropdown with:', json.zones);
  } catch (err) {
    console.error("Error fetching zones:", err);
  }
}

async function populateCameras() {
  try {
    console.log('Fetching production cameras from:', "/api/production_cameras");
    const res = await fetch("/api/production_cameras");
    const json = await res.json();
    console.log('Production cameras response:', json);
    if (!json.ok) return;

    const select = document.getElementById("camera");
    select.innerHTML = '<option>All Cameras</option>';

    json.cameras.forEach((camera) => {
      const opt = document.createElement("option");
      opt.value = camera;
      opt.text = camera;
      select.appendChild(opt);
    });
    console.log('Populated cameras dropdown with:', json.cameras);
  } catch (err) {
    console.error("Error fetching cameras:", err);
  }
}

// ==================== Fetch Data ====================

async function fetchTable(params) {
  const url = new URL("/api/production_data", location.origin);
  params.page = currentPage;
  params.per_page = 50;
  Object.entries(params).forEach(([k, v]) => v && url.searchParams.append(k, v));
  console.log('Fetching table data from:', url.toString());
  return await getJSON(url);
}

async function fetchTimeSeries(params) {
  const url = new URL("/api/production_time_series", location.origin);
  Object.entries(params).forEach(([k, v]) => v && url.searchParams.append(k, v));
  console.log('Fetching time series data from:', url.toString());
  return await getJSON(url);
}

async function fetchHourly(params) {
  const url = new URL("/api/production_hourly", location.origin);
  Object.entries(params).forEach(([k, v]) => v && url.searchParams.append(k, v));
  console.log('Fetching hourly data from:', url.toString());
  return await getJSON(url);
}

// ==================== Populate Table ====================

function populateTable(data) {
  const tbody = document.querySelector("#alerts-table tbody");
  tbody.innerHTML = "";

  if (!data || !data.length) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:#888;">No data available</td></tr>`;
    return;
  }

  data.forEach((doc) => {
    const row = `
      <tr>
        <td>${doc.date_time || "-"}</td>
        <td>${doc.alert_type || "-"}</td>
        <td>${doc.camera_no || "-"}</td>
        <td>${doc.area || "-"}</td>
        <td>${
          doc.image_byte_str
            ? `<a href="${doc.image_byte_str}" target="_blank">View</a>`
            : "-"
        }</td>
      </tr>`;
    tbody.insertAdjacentHTML("beforeend", row);
  });
}

// ==================== Chart.js Config ====================

let trendChart = null;
let hourChart = null;

// ==================== Render Charts ====================

function renderTrend(dates, types, series) {
  const ctx = document.getElementById("trendChart").getContext("2d");
  if (trendChart) {
    trendChart.destroy();
    trendChart = null;
  }

  // Calculate total count
  let totalCount = 0;
  types.forEach(t => {
    const seriesData = series[t] || [];
    totalCount += seriesData.reduce((sum, val) => sum + val, 0);
  });

  const datasets = types.map((t, i) => ({
    label: t,
    data: series[t] || [],
    borderColor: CHART_COLORS[i % CHART_COLORS.length],
    backgroundColor: CHART_COLORS[i % CHART_COLORS.length],
    tension: 0.1,
    borderWidth: 2,
    pointRadius: 3,
    fill: false,
  }));

  trendChart = new Chart(ctx, {
    type: "line",
    data: { labels: dates, datasets },
    options: {
      ...BASE_CHART_OPTIONS,
      plugins: {
        ...BASE_CHART_OPTIONS.plugins,
        title: {
          display: true,
          text: `Total Alerts: ${totalCount}`,
          align: 'end',
          font: {
            size: 14,
            weight: 'bold'
          },
          padding: {
            top: 0,
            bottom: 10
          }
        }
      },
      scales: {
        ...BASE_CHART_OPTIONS.scales,
        x: { ...BASE_CHART_OPTIONS.scales.x, title: { display: true, text: "Date" } },
        y: { ...BASE_CHART_OPTIONS.scales.y, title: { display: true, text: "Alerts" } },
      },
    },
  });

  trendChart.stop();
}

function renderHourly(hours, types, series) {
  const ctx = document.getElementById("hourChart").getContext("2d");
  if (hourChart) {
    hourChart.destroy();
    hourChart = null;
  }

  const colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf"
  ];

  const datasets = types.map((t, i) => ({
    label: t,
    data: series[t] || [],
    backgroundColor: colors[i % colors.length],
  }));

  hourChart = new Chart(ctx, {
    type: "bar",
    data: { labels: hours.map((h) => `${h}:00`), datasets },
    options: {
      ...BASE_CHART_OPTIONS,
      scales: {
        ...BASE_CHART_OPTIONS.scales,
        x: { stacked: true, title: { display: true, text: "Hour of Day" } },
        y: { stacked: true, title: { display: true, text: "Alerts" } },
      },
    },
  });

  hourChart.stop();
}

// === Filter Logic ===
async function applyFilters() {
  const s = document.getElementById("start-date").value;
  const e = document.getElementById("end-date").value;
  const z = document.getElementById("area").value;
  const l = document.getElementById("line").value;
  const zn = document.getElementById("zone").value;
  const cam = document.getElementById("camera").value;
  const at = document.getElementById("alert-type").value;
  const sh = document.getElementById("shift").value;

  const params = { start_date: s, end_date: e, area: z, line: l, zone: zn, camera: cam, alert_type: at, shift: sh };
  
  console.log("Applying filters with params:", params);

  try {
    const [table, ts, hr] = await Promise.all([
      fetchTable(params),
      fetchTimeSeries(params),
      fetchHourly(params),
    ]);

    console.log("API Responses:", { table, ts, hr });

    if (table && table.ok) {
      console.log("Table data:", table.data);
      populateTable(table.data || []);
      updatePaginationInfo(table.page, table.per_page, table.total);
      renderPagination(table.page, table.total_pages);
    } else {
      console.error("Table API failed:", table);
    }
    
    if (ts && ts.ok) {
      console.log("Time series data:", ts);
      renderTrend(ts.dates, ts.types, ts.series);
    } else {
      console.error("Time series API failed:", ts);
    }
    
    if (hr && hr.ok) {
      console.log("Hourly data:", hr);
      renderHourly(hr.hours, hr.types, hr.series);
    } else {
      console.error("Hourly API failed:", hr);
    }
  } catch (err) {
    console.error("Error fetching data:", err);
  }
}

// === Event Listeners ===
document.getElementById("apply-filters").addEventListener("click", () => {
  currentPage = 1;
  applyFilters();
});
document.getElementById("clear-filters").addEventListener("click", clearFilters);

// === Initialize ===
(async function init() {
  setDefaultDates();
  setupPaginationListeners();
  await populateLines();
  await populateZones();
  await populateCameras();
  await applyFilters();

  window.addEventListener("resize", () => {
    if (trendChart) trendChart.resize();
    if (hourChart) hourChart.resize();
  });
})();