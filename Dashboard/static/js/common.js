// ==================== Shared Utility Functions ====================

let currentPage = 1;
let totalPages = 1;

function formatDate(d) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function setDefaultDates() {
  const end = new Date();
  const start = new Date();
  start.setDate(end.getDate() - 14);
  document.getElementById("start-date").value = formatDate(start);
  document.getElementById("end-date").value = formatDate(end);
}

async function getJSON(url) {
  const res = await fetch(url);
  return await res.json();
}

function clearFilters() {
  setDefaultDates();
  const area = document.getElementById("area");
  const line = document.getElementById("line");
  const shift = document.getElementById("shift");
  const alertType = document.getElementById("alert-type");
  const plant = document.getElementById("plant");
  
  if (area) area.value = "All Areas";
  if (line) line.value = "All Lines";
  if (shift) shift.value = "All Shifts";
  if (alertType) alertType.value = "All Alert Types";
  if (plant) plant.value = "All Plants";
  
  currentPage = 1;
  applyFilters();
}

function updatePaginationInfo(page, perPage, total) {
  const start = total === 0 ? 0 : (page - 1) * perPage + 1;
  const end = Math.min(page * perPage, total);
  document.getElementById("pagination-info").textContent = `Showing ${start}-${end} of ${total} results`;
}

function renderPagination(page, totalPagesCount) {
  currentPage = page;
  totalPages = totalPagesCount;
  
  const prevBtn = document.getElementById("prev-page");
  const nextBtn = document.getElementById("next-page");
  const pageNumbers = document.getElementById("page-numbers");
  
  prevBtn.disabled = page === 1;
  nextBtn.disabled = page === totalPagesCount || totalPagesCount === 0;
  
  // Generate page numbers
  let pagesHTML = "";
  const maxVisible = 5;
  let startPage = Math.max(1, page - Math.floor(maxVisible / 2));
  let endPage = Math.min(totalPagesCount, startPage + maxVisible - 1);
  
  if (endPage - startPage < maxVisible - 1) {
    startPage = Math.max(1, endPage - maxVisible + 1);
  }
  
  if (startPage > 1) {
    pagesHTML += '<span class="page-num" data-page="1">1</span>';
    if (startPage > 2) pagesHTML += '<span style="padding: 0 5px;">...</span>';
  }
  
  for (let i = startPage; i <= endPage; i++) {
    pagesHTML += `<span class="page-num ${i === page ? 'active' : ''}" data-page="${i}">${i}</span>`;
  }
  
  if (endPage < totalPagesCount) {
    if (endPage < totalPagesCount - 1) pagesHTML += '<span style="padding: 0 5px;">...</span>';
    pagesHTML += `<span class="page-num" data-page="${totalPagesCount}">${totalPagesCount}</span>`;
  }
  
  pageNumbers.innerHTML = pagesHTML;
  
  // Add click listeners to page numbers
  document.querySelectorAll(".page-num").forEach(btn => {
    btn.addEventListener("click", () => {
      const pageNum = parseInt(btn.dataset.page);
      if (pageNum !== currentPage) {
        currentPage = pageNum;
        applyFilters();
      }
    });
  });
}

function setupPaginationListeners() {
  document.getElementById("prev-page").addEventListener("click", () => {
    if (currentPage > 1) {
      currentPage--;
      applyFilters();
    }
  });
  
  document.getElementById("next-page").addEventListener("click", () => {
    if (currentPage < totalPages) {
      currentPage++;
      applyFilters();
    }
  });
}

// ==================== Shared Chart Configuration ====================

const CHART_COLORS = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
  "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
  "#bcbd22", "#17becf"
];

const BASE_CHART_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  interaction: {
    intersect: false,
  },
  plugins: {
    legend: {
      position: "bottom",
      labels: { font: { size: 12 }, color: "#333" },
    },
  },
  scales: {
    x: {
      grid: { color: "#eee" },
      ticks: { color: "#555" },
    },
    y: {
      beginAtZero: true,
      grid: { color: "#eee" },
      ticks: { color: "#555" },
    },
  },
};

// ==================== UI Animations / Tab Navigation ====================

function animateAndNavigate(elem, href) {
  try {
    elem.classList.add('fade-out');
    // small delay for animation then navigate
    setTimeout(() => { window.location.href = href; }, 220);
  } catch (e) { window.location.href = href; }
}

function setupNavAnimations() {
  // animated navigation for sidebar items that have data-href
  document.querySelectorAll('.nav-item[data-href]').forEach(btn => {
    btn.addEventListener('click', (ev) => {
      const href = btn.dataset.href;
      
      // Remove active classes from all items
      document.querySelectorAll('.nav-item').forEach(nav => {
        nav.classList.remove('active');
      });
      
      // Add active class to clicked item
      btn.classList.add('active');
      
      // animate main content out then navigate
      const main = document.querySelector('.content');
      if (main) main.classList.add('fade-out');
      animateAndNavigate(main || btn, href);
    });
  });

  // clickable alerts card
  const alertsCards = document.querySelectorAll('.alerts-card');
  alertsCards.forEach(alertsCard => {
    if (alertsCard) {
      alertsCard.addEventListener('click', (e) => {
        e.preventDefault();
        alertsCard.classList.add('pulse');
        setTimeout(() => {
          const href = alertsCard.getAttribute('href') || alertsCard.dataset.href || '/alerts';
          window.location.href = href;
        }, 380);
      });
    }
  });
}

// initialize small UI behavior on load
document.addEventListener('DOMContentLoaded', function() {
  try { setupNavAnimations(); } catch (e) { /* ignore */ }
});
