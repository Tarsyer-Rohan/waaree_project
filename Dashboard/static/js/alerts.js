// === Populate Dynamic Filters ===
async function populateLines() {
  try {
    // Get section from URL
    const urlParams = new URLSearchParams(location.search);
    const section = urlParams.get('section') || 'strategic';
    
    const select = document.getElementById("line");
    if (!select) return; // Skip if line filter doesn't exist
    
    const res = await fetch(`/api/alerts_lines?section=${section}`);
    const json = await res.json();
    if (!json.ok) return;
    select.innerHTML = '<option>All Lines</option>';
    json.lines.forEach((line) => {
      const opt = document.createElement("option");
      opt.value = line;
      opt.text = `Line ${line}`;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error("Error fetching lines:", err);
  }
}

async function populateZones() {
  try {
    // Get section from URL
    const urlParams = new URLSearchParams(location.search);
    const section = urlParams.get('section') || 'strategic';
    
    const select = document.getElementById("zone");
    if (!select) return; // Skip if zone filter doesn't exist
    
    const apiUrl = section === 'compliance' ? '/api/compliance_zones' : '/api/compliance_zones';
    const res = await fetch(apiUrl);
    const json = await res.json();
    if (!json.ok) return;
    select.innerHTML = '<option>All Zones</option>';
    json.zones.forEach((zone) => {
      const opt = document.createElement("option");
      opt.value = zone;
      opt.text = zone;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error("Error fetching zones:", err);
  }
}

async function populateCameras() {
  try {
    // Get section from URL
    const urlParams = new URLSearchParams(location.search);
    const section = urlParams.get('section') || 'strategic';
    
    const select = document.getElementById("camera");
    if (!select) return; // Skip if camera filter doesn't exist
    
    const apiUrl = section === 'compliance' ? '/api/compliance_cameras' : '/api/strategic_cameras';
    const res = await fetch(apiUrl);
    const json = await res.json();
    if (!json.ok) return;
    select.innerHTML = '<option>All Cameras</option>';
    json.cameras.forEach((camera) => {
      const opt = document.createElement("option");
      opt.value = camera;
      opt.text = camera;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error("Error fetching cameras:", err);
  }
}

// === Fetch APIs ===
async function fetchTable(params) {
  const url = new URL("/api/alerts_data", location.origin);
  // Add section parameter from URL
  const urlParams = new URLSearchParams(location.search);
  const section = urlParams.get('section') || 'strategic';
  params.section = section;
  params.page = currentPage;
  params.per_page = 50;
  
  Object.entries(params).forEach(([k, v]) => v && url.searchParams.append(k, v));
  return await getJSON(url);
}

async function fetchTimeSeries(params) {
  const url = new URL("/api/alerts_time_series", location.origin);
  // Add section parameter from URL
  const urlParams = new URLSearchParams(location.search);
  const section = urlParams.get('section') || 'strategic';
  params.section = section;
  
  Object.entries(params).forEach(([k, v]) => v && url.searchParams.append(k, v));
  return await getJSON(url);
}

async function fetchHourly(params) {
  const url = new URL("/api/alerts_hourly", location.origin);
  // Add section parameter from URL
  const urlParams = new URLSearchParams(location.search);
  const section = urlParams.get('section') || 'strategic';
  params.section = section;
  
  Object.entries(params).forEach(([k, v]) => v && url.searchParams.append(k, v));
  return await getJSON(url);
}

// === Populate Table ===
function populateTable(data) {
  const tbody = document.querySelector("#alerts-table tbody");
  tbody.innerHTML = "";
  if (!data || !data.length) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:#999;">No data found</td></tr>`;
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

// === Chart.js Configuration ===
let trendChart = null;
let hourChart = null;

// === Render Trend Line Chart ===
function renderTrend(dates, types, series) {
  const ctx = document.getElementById("trendChart").getContext("2d");
  if (trendChart) trendChart.destroy();

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
    fill: false,
    tension: 0.3,
    borderWidth: 2,
    pointRadius: 3,
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
        y: { ...BASE_CHART_OPTIONS.scales.y, title: { display: true, text: "Count" } },
      },
    },
  });
}

// === Render Hourly Stacked Bar Chart ===
function renderHourly(hours, types, series) {
  const ctx = document.getElementById("hourChart").getContext("2d");
  if (hourChart) hourChart.destroy();

  const datasets = types.map((t, i) => ({
    label: t,
    data: series[t] || [],
    backgroundColor: CHART_COLORS[i % CHART_COLORS.length],
  }));

  hourChart = new Chart(ctx, {
    type: "bar",
    data: { labels: hours.map((h) => `${h}:00`), datasets },
    options: {
      ...BASE_CHART_OPTIONS,
      scales: {
        ...BASE_CHART_OPTIONS.scales,
        x: { stacked: true, title: { display: true, text: "Hour of Day" } },
        y: { ...BASE_CHART_OPTIONS.scales.y, stacked: true },
      },
    },
  });
}

// === Filter Logic ===
async function applyFilters() {
  const s = document.getElementById("start-date").value;
  const e = document.getElementById("end-date").value;
  const z = document.getElementById("area").value;
  const zn = document.getElementById("zone")?.value;
  const cam = document.getElementById("camera")?.value;
  const at = document.getElementById("alert-type").value;
  const sh = document.getElementById("shift").value;

  const params = { start_date: s, end_date: e, area: z, zone: zn, camera: cam, alert_type: at, shift: sh };

  try {
    const [table, ts, hr] = await Promise.all([
      fetchTable(params),
      fetchTimeSeries(params),
      fetchHourly(params),
    ]);

    if (table.ok) {
      populateTable(table.data || []);
      updatePaginationInfo(table.page, table.per_page, table.total);
      renderPagination(table.page, table.total_pages);
    }
    if (ts.ok) renderTrend(ts.dates, ts.types, ts.series);
    if (hr.ok) renderHourly(hr.hours, hr.types, hr.series);
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
})();
