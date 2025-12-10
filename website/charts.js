document.addEventListener('DOMContentLoaded', function() {
    // Shared Chart Defaults
    Chart.defaults.color = '#94A3B8';
    Chart.defaults.font.family = "'Inter', system-ui, -apple-system, sans-serif";
    Chart.defaults.borderColor = '#334155';

    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    padding: 20,
                    usePointStyle: true
                }
            },
            tooltip: {
                backgroundColor: '#1E293B',
                titleColor: '#F1F5F9',
                bodyColor: '#94A3B8',
                borderColor: '#334155',
                borderWidth: 1,
                padding: 12,
                cornerRadius: 8,
                displayColors: true
            }
        },
        scales: {
            y: {
                grid: {
                    color: '#334155',
                    borderDash: [5, 5]
                }
            },
            x: {
                grid: {
                    display: false
                }
            }
        }
    };

    // Validation Loss Chart
    const ctxLoss = document.getElementById('lossChart').getContext('2d');
    new Chart(ctxLoss, {
        type: 'line',
        data: {
            labels: ['Epoch 1', 'Epoch 2', 'Epoch 3'],
            datasets: [
                {
                    label: 'Single-View Baseline',
                    data: [3.0287, 3.1253, 3.2278],
                    borderColor: '#3B82F6', // Blue
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Multi-View Baseline',
                    data: [3.2196, 3.1427, 2.9233],
                    borderColor: '#10B981', // Green
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            ...commonOptions,
            plugins: {
                ...commonOptions.plugins,
                title: {
                    display: true,
                    text: 'Validation Loss (Lower is Better)',
                    font: { size: 16, weight: '600' },
                    color: '#F1F5F9'
                }
            }
        }
    });

    // Foul Type Accuracy Chart
    const ctxAcc = document.getElementById('accChart').getContext('2d');
    new Chart(ctxAcc, {
        type: 'line',
        data: {
            labels: ['Epoch 1', 'Epoch 2', 'Epoch 3'],
            datasets: [
                {
                    label: 'Single-View Baseline',
                    data: [0.0855, 0.0884, 0.0900],
                    borderColor: '#3B82F6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Multi-View Baseline',
                    data: [0.0810, 0.0928, 0.1091],
                    borderColor: '#10B981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            ...commonOptions,
            plugins: {
                ...commonOptions.plugins,
                title: {
                    display: true,
                    text: 'Foul Type Balanced Accuracy (Higher is Better)',
                    font: { size: 16, weight: '600' },
                    color: '#F1F5F9'
                }
            }
        }
    });
});
