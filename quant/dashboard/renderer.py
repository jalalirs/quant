"""
Dashboard HTML renderer for benchmark comparison
"""

import json
import os
import logging
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from jinja2 import Template

logger = logging.getLogger(__name__)

class DashboardRenderer:
    """Generate HTML dashboard for benchmark comparison"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dashboard_config = config.get('dashboard', {})
        self.title = self.dashboard_config.get('title', 'Model Performance Dashboard')
        self.output_file = self.dashboard_config.get('output_file', 'results/dashboard.html')
        self.theme = self.dashboard_config.get('theme', {})
        
        # Benchmark data
        self.benchmark_data = []
        self.summary_stats = {}
        
        # Create output directory
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
    
    @classmethod
    def load_from_dict(cls, config: Dict[str, Any]) -> 'DashboardRenderer':
        """Load dashboard renderer from configuration"""
        return cls(config)
    
    def load_benchmark_files(self) -> bool:
        """Load all benchmark result files"""
        benchmark_files = self.dashboard_config.get('benchmark_files', [])
        
        for file_config in benchmark_files:
            file_path = file_config.get('path')
            label = file_config.get('label')
            color = file_config.get('color', '#3498DB')
            description = file_config.get('description', '')
            
            if not os.path.exists(file_path):
                logger.warning(f"Benchmark file not found: {file_path}")
                continue
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Add metadata
                data['_metadata'] = {
                    'label': label,
                    'color': color,
                    'description': description,
                    'file_path': file_path
                }
                
                self.benchmark_data.append(data)
                logger.info(f"Loaded benchmark data: {label}")
                
            except Exception as e:
                logger.error(f"Failed to load benchmark file {file_path}: {e}")
        
        return len(self.benchmark_data) > 0
    
    def calculate_summary_stats(self):
        """Calculate summary statistics across all benchmarks"""
        if not self.benchmark_data:
            return
        
        self.summary_stats = {
            'total_models': len(self.benchmark_data),
            'models': [],
            'best_latency': {'model': '', 'value': float('inf')},
            'best_throughput': {'model': '', 'value': 0},
            'highest_success_rate': {'model': '', 'value': 0},
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        for data in self.benchmark_data:
            metadata = data['_metadata']
            stats = data.get('statistics', {})
            latency_stats = stats.get('latency_stats', {})
            throughput_stats = stats.get('throughput_stats', {})
            
            model_summary = {
                'label': metadata['label'],
                'color': metadata['color'],
                'description': metadata['description'],
                'total_requests': stats.get('total_requests', 0),
                'success_rate': stats.get('success_rate', 0),
                'mean_latency': latency_stats.get('mean', 0),
                'mean_throughput': throughput_stats.get('mean_tokens_per_second', 0),
                'p95_latency': latency_stats.get('p95', 0)
            }
            
            self.summary_stats['models'].append(model_summary)
            
            # Track best performers
            mean_latency = latency_stats.get('mean', float('inf'))
            if mean_latency < self.summary_stats['best_latency']['value']:
                self.summary_stats['best_latency'] = {
                    'model': metadata['label'],
                    'value': mean_latency
                }
            
            mean_throughput = throughput_stats.get('mean_tokens_per_second', 0)
            if mean_throughput > self.summary_stats['best_throughput']['value']:
                self.summary_stats['best_throughput'] = {
                    'model': metadata['label'],
                    'value': mean_throughput
                }
            
            success_rate = stats.get('success_rate', 0)
            if success_rate > self.summary_stats['highest_success_rate']['value']:
                self.summary_stats['highest_success_rate'] = {
                    'model': metadata['label'],
                    'value': success_rate
                }
    
    def generate_chart_data(self) -> Dict[str, Any]:
        """Generate data for all charts"""
        chart_data = {}
        
        # Latency comparison data
        latency_data = {
            'labels': [],
            'datasets': [{
                'label': 'Mean Latency (s)',
                'data': [],
                'backgroundColor': [],
                'p95': [],
                'p99': []
            }]
        }
        
        # Throughput comparison data
        throughput_data = {
            'labels': [],
            'datasets': [{
                'label': 'Tokens per Second',
                'data': [],
                'backgroundColor': []
            }]
        }
        
        # Success rate data
        success_data = {
            'labels': [],
            'datasets': [{
                'label': 'Success Rate (%)',
                'data': [],
                'backgroundColor': []
            }]
        }
        
        # Latency distribution data (for box plot)
        distribution_data = {
            'labels': [],
            'datasets': []
        }
        
        for data in self.benchmark_data:
            metadata = data['_metadata']
            stats = data.get('statistics', {})
            detailed_results = data.get('detailed_results', [])
            
            label = metadata['label']
            color = metadata['color']
            
            # Extract metrics
            latency_stats = stats.get('latency_stats', {})
            throughput_stats = stats.get('throughput_stats', {})
            success_rate = stats.get('success_rate', 0)
            
            # Latency data
            latency_data['labels'].append(label)
            latency_data['datasets'][0]['data'].append(latency_stats.get('mean', 0))
            latency_data['datasets'][0]['backgroundColor'].append(color)
            latency_data['datasets'][0]['p95'].append(latency_stats.get('p95', 0))
            latency_data['datasets'][0]['p99'].append(latency_stats.get('p99', 0))
            
            # Throughput data
            throughput_data['labels'].append(label)
            throughput_data['datasets'][0]['data'].append(throughput_stats.get('mean_tokens_per_second', 0))
            throughput_data['datasets'][0]['backgroundColor'].append(color)
            
            # Success rate data
            success_data['labels'].append(label)
            success_data['datasets'][0]['data'].append(success_rate)
            success_data['datasets'][0]['backgroundColor'].append(color)
            
            # Distribution data (latency values for each model)
            latencies = [result['latency'] for result in detailed_results if result.get('success', False)]
            if latencies:
                distribution_data['labels'].append(label)
                # Calculate quartiles for box plot
                sorted_latencies = sorted(latencies)
                n = len(sorted_latencies)
                quartiles = {
                    'min': min(latencies),
                    'q1': sorted_latencies[int(n * 0.25)],
                    'median': sorted_latencies[int(n * 0.5)],
                    'q3': sorted_latencies[int(n * 0.75)],
                    'max': max(latencies)
                }
                
                if 'datasets' not in distribution_data:
                    distribution_data['datasets'] = []
                
                distribution_data['datasets'].append({
                    'label': label,
                    'backgroundColor': color,
                    'data': [quartiles],
                    'raw_data': latencies[:100]  # Limit for performance
                })
        
        chart_data = {
            'latency_comparison': latency_data,
            'throughput_comparison': throughput_data,
            'success_rate': success_data,
            'latency_distribution': distribution_data
        }
        
        return chart_data
    
    def generate_html(self) -> str:
        """Generate the complete HTML dashboard"""
        chart_data = self.generate_chart_data()
        
        # HTML template
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: {{ theme.font_family or 'Inter, sans-serif' }};
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: {{ theme.text_color or '#2C3E50' }};
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .header h1 {
            color: {{ theme.primary_color or '#2C3E50' }};
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 700;
        }
        
        .header p {
            color: #7f8c8d;
            font-size: 1.1rem;
            max-width: 600px;
            margin: 0 auto;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-card h3 {
            color: {{ theme.primary_color or '#2C3E50' }};
            font-size: 2rem;
            margin-bottom: 5px;
            font-weight: 600;
        }
        
        .stat-card p {
            color: #7f8c8d;
            font-size: 0.9rem;
        }
        
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .chart-container {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .chart-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 15px;
            color: {{ theme.primary_color or '#2C3E50' }};
        }
        
        .chart-description {
            color: #7f8c8d;
            font-size: 0.9rem;
            margin-bottom: 20px;
        }
        
        .model-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 20px;
            justify-content: center;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            background: #f8f9fa;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
        }
        
        .legend-color {
            width: 16px;
            height: 16px;
            border-radius: 50%;
        }
        
        .summary-table {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow-x: auto;
        }
        
        .summary-table h2 {
            color: {{ theme.primary_color or '#2C3E50' }};
            margin-bottom: 20px;
            font-size: 1.5rem;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            text-align: left;
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }
        
        th {
            background: #f8f9fa;
            font-weight: 600;
            color: {{ theme.primary_color or '#2C3E50' }};
        }
        
        tr:hover {
            background: #f8f9fa;
        }
        
        .model-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: #f8f9fa;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.85rem;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: white;
            font-size: 0.9rem;
        }
        
        @media (max-width: 768px) {
            .chart-grid {
                grid-template-columns: 1fr;
            }
            
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .header h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>{{ title }}</h1>
            <p>{{ description }}</p>
            <p><small>Generated on {{ summary_stats.generation_time }}</small></p>
        </div>
        
        <!-- Summary Stats -->
        <div class="stats-grid">
            <div class="stat-card">
                <h3>{{ summary_stats.total_models }}</h3>
                <p>Models Compared</p>
            </div>
            <div class="stat-card">
                <h3>{{ "%.3f"|format(summary_stats.best_latency.value) }}s</h3>
                <p>Best Latency<br><small>{{ summary_stats.best_latency.model }}</small></p>
            </div>
            <div class="stat-card">
                <h3>{{ "%.1f"|format(summary_stats.best_throughput.value) }}</h3>
                <p>Best Throughput (tok/s)<br><small>{{ summary_stats.best_throughput.model }}</small></p>
            </div>
            <div class="stat-card">
                <h3>{{ "%.1f"|format(summary_stats.highest_success_rate.value) }}%</h3>
                <p>Highest Success Rate<br><small>{{ summary_stats.highest_success_rate.model }}</small></p>
            </div>
        </div>
        
        <!-- Model Legend -->
        <div class="chart-container">
            <div class="chart-title">Model Configurations</div>
            <div class="model-legend">
                {% for model in summary_stats.models %}
                <div class="legend-item">
                    <div class="legend-color" style="background: {{ model.color }}"></div>
                    <span>{{ model.label }}</span>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- Charts -->
        <div class="chart-grid">
            <!-- Latency Comparison -->
            <div class="chart-container">
                <div class="chart-title">Response Latency Comparison</div>
                <div class="chart-description">Average response latency with percentile markers</div>
                <canvas id="latencyChart" width="400" height="300"></canvas>
            </div>
            
            <!-- Throughput Comparison -->
            <div class="chart-container">
                <div class="chart-title">Token Generation Throughput</div>
                <div class="chart-description">Tokens generated per second</div>
                <canvas id="throughputChart" width="400" height="300"></canvas>
            </div>
            
            <!-- Success Rate -->
            <div class="chart-container">
                <div class="chart-title">Success Rate Analysis</div>
                <div class="chart-description">Request success rates across models</div>
                <canvas id="successChart" width="400" height="300"></canvas>
            </div>
            
            <!-- Latency Distribution -->
            <div class="chart-container" style="grid-column: 1 / -1;">
                <div class="chart-title">Latency Distribution</div>
                <div class="chart-description">Distribution of response latencies across all requests</div>
                <canvas id="distributionChart" width="800" height="400"></canvas>
            </div>
        </div>
        
        <!-- Summary Table -->
        <div class="summary-table">
            <h2>Detailed Performance Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Total Requests</th>
                        <th>Success Rate</th>
                        <th>Mean Latency (s)</th>
                        <th>P95 Latency (s)</th>
                        <th>Mean Throughput (tok/s)</th>
                    </tr>
                </thead>
                <tbody>
                    {% for model in summary_stats.models %}
                    <tr>
                        <td>
                            <div class="model-badge">
                                <div class="legend-color" style="background: {{ model.color }}"></div>
                                {{ model.label }}
                            </div>
                        </td>
                        <td>{{ model.total_requests }}</td>
                        <td>{{ "%.1f"|format(model.success_rate) }}%</td>
                        <td>{{ "%.3f"|format(model.mean_latency) }}</td>
                        <td>{{ "%.3f"|format(model.p95_latency) }}</td>
                        <td>{{ "%.1f"|format(model.mean_throughput) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
    
    <div class="footer">
        Generated by Quantization Pipeline Dashboard • {{ summary_stats.generation_time }}
    </div>

    <script>
        // Chart.js configuration
        Chart.defaults.font.family = '{{ theme.font_family or "Inter, sans-serif" }}';
        Chart.defaults.color = '{{ theme.text_color or "#2C3E50" }}';
        
        const chartData = {{ chart_data | tojson }};
        
        // Latency Chart
        new Chart(document.getElementById('latencyChart'), {
            type: 'bar',
            data: chartData.latency_comparison,
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            afterBody: function(context) {
                                const index = context[0].dataIndex;
                                const dataset = context[0].dataset;
                                return [
                                    `P95: ${dataset.p95[index].toFixed(3)}s`,
                                    `P99: ${dataset.p99[index].toFixed(3)}s`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Latency (seconds)'
                        }
                    }
                }
            }
        });
        
        // Throughput Chart
        new Chart(document.getElementById('throughputChart'), {
            type: 'bar',
            data: chartData.throughput_comparison,
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Tokens per Second'
                        }
                    }
                }
            }
        });
        
        // Success Rate Chart
        new Chart(document.getElementById('successChart'), {
            type: 'doughnut',
            data: {
                labels: chartData.success_rate.labels,
                datasets: [{
                    data: chartData.success_rate.datasets[0].data,
                    backgroundColor: chartData.success_rate.datasets[0].backgroundColor,
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.parsed.toFixed(1)}%`;
                            }
                        }
                    }
                }
            }
        });
        
        // Distribution Chart (Violin plot approximation)
        const distributionCtx = document.getElementById('distributionChart');
        new Chart(distributionCtx, {
            type: 'boxplot',
            data: {
                labels: chartData.latency_distribution.labels,
                datasets: chartData.latency_distribution.datasets.map(dataset => ({
                    label: dataset.label,
                    backgroundColor: dataset.backgroundColor + '40',
                    borderColor: dataset.backgroundColor,
                    borderWidth: 2,
                    data: dataset.data
                }))
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'Latency (seconds)'
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>
        """
        
        # Render template
        template = Template(template_str)
        html_content = template.render(
            title=self.title,
            description=self.dashboard_config.get('description', ''),
            summary_stats=self.summary_stats,
            chart_data=chart_data,
            theme=self.theme
        )
        
        return html_content
    
    def render(self) -> bool:
        """Render the complete dashboard"""
        logger.info("Generating dashboard...")
        
        # Load benchmark files
        if not self.load_benchmark_files():
            logger.error("No benchmark files loaded")
            return False
        
        # Calculate summary statistics
        self.calculate_summary_stats()
        
        # Generate HTML
        html_content = self.generate_html()
        
        # Write to file
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Dashboard generated successfully: {self.output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write dashboard file: {e}")
            return False