document.addEventListener('DOMContentLoaded', function() {
    const dailyChartCtx = document.getElementById('dailyChart').getContext('2d');
    const hourlyChartCtx = document.getElementById('hourlyChart').getContext('2d');
    let dailyChart, hourlyChart;

    function loadCameras() {
        fetch('/api/footfall_cameras')
            .then(res => res.json())
            .then(data => {
                if (data.ok && data.cameras) {
                    const camSelect = document.getElementById('cam_no');
                    // Keep "All Cameras" option
                    camSelect.innerHTML = '<option value="All Cameras">All Cameras</option>';
                    // Add camera options
                    data.cameras.forEach(cam => {
                        const option = document.createElement('option');
                        option.value = cam;
                        option.textContent = cam;
                        camSelect.appendChild(option);
                    });
                }
            })
            .catch(err => console.error('Error loading cameras:', err));
    }

    function fetchAndRender() {
        const start_date = document.getElementById('start_date').value;
        const end_date = document.getElementById('end_date').value;
        const cam_no = document.getElementById('cam_no').value;
        let params = `?start_date=${start_date}&end_date=${end_date}`;
        if (cam_no && cam_no !== 'All Cameras') params += `&cam_no=${cam_no}`;

        fetch(`/api/footfall_daily${params}`)
            .then(res => res.json())
            .then(data => {
                if (dailyChart) dailyChart.destroy();
                const totalCount = data.counts ? data.counts.reduce((sum, val) => sum + val, 0) : 0;
                dailyChart = new Chart(dailyChartCtx, {
                    type: 'line',
                    data: {
                        labels: data.dates,
                        datasets: [{ label: 'Footfall', data: data.counts, borderColor: '#007bff', backgroundColor: 'rgba(0,123,255,0.1)', fill: true, tension: 0.4 }]
                    },
                    options: { 
                        responsive: true, 
                        maintainAspectRatio: false, 
                        plugins: { 
                            legend: { display: true },
                            title: {
                                display: true,
                                text: `Total Footfall: ${totalCount}`,
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
                        } 
                    }
                });
            });
        fetch(`/api/footfall_hourly${params}`)
            .then(res => res.json())
            .then(data => {
                if (hourlyChart) hourlyChart.destroy();
                hourlyChart = new Chart(hourlyChartCtx, {
                    type: 'line',
                    data: {
                        labels: data.hours.map(h => h+':00'),
                        datasets: [{ label: 'Footfall', data: data.counts, borderColor: '#28a745', backgroundColor: 'rgba(40,167,69,0.1)', fill: true, tension: 0.4 }]
                    },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: true } } }
                });
            });
    }
    
    document.getElementById('apply-filters').addEventListener('click', function() {
        fetchAndRender();
    });
    
    document.getElementById('clear-filters').addEventListener('click', function() {
        document.getElementById('start_date').value = '';
        document.getElementById('end_date').value = '';
        document.getElementById('cam_no').value = 'All Cameras';
        fetchAndRender();
    });
    
    loadCameras();
    fetchAndRender();
});
