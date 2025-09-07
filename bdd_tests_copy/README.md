<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BDD Test Automation Framework - Architecture</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 30px;
            overflow: hidden;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .header .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .architecture-container {
            display: grid;
            grid-template-columns: 1fr;
            gap: 30px;
        }

        /* Main Architecture Diagram */
        .main-diagram {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            border: 2px solid #e9ecef;
        }

        .diagram-title {
            text-align: center;
            font-size: 1.8em;
            color: #2c3e50;
            margin-bottom: 30px;
            font-weight: 600;
        }

        .architecture-layers {
            display: grid;
            grid-template-rows: repeat(6, auto);
            gap: 20px;
        }

        .layer {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            border-left: 5px solid;
            position: relative;
        }

        .layer.presentation { border-left-color: #3498db; }
        .layer.test-execution { border-left-color: #e74c3c; }
        .layer.api-client { border-left-color: #f39c12; }
        .layer.data-management { border-left-color: #27ae60; }
        .layer.infrastructure { border-left-color: #9b59b6; }
        .layer.reporting { border-left-color: #1abc9c; }

        .layer-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 15px;
        }

        .layer-title {
            font-size: 1.3em;
            font-weight: 600;
            color: #2c3e50;
        }

        .layer-badge {
            background: #667eea;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: 500;
        }

        .layer-components {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }

        .component {
            background: #f1f3f4;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
        }

        .component:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            background: #e8f4fd;
        }

        .component-name {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 8px;
            font-size: 1em;
        }

        .component-tech {
            font-size: 0.85em;
            color: #7f8c8d;
            font-style: italic;
        }

        .component-description {
            font-size: 0.8em;
            color: #555;
            margin-top: 5px;
            line-height: 1.3;
        }

        /* Flow Arrows */
        .flow-arrow {
            position: absolute;
            right: 20px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 2em;
            color: #667eea;
            opacity: 0.7;
        }

        /* Data Flow Section */
        .data-flow {
            background: white;
            border-radius: 15px;
            padding: 30px;
            border: 2px solid #e9ecef;
            margin-top: 30px;
        }

        .flow-title {
            text-align: center;
            font-size: 1.6em;
            color: #2c3e50;
            margin-bottom: 25px;
            font-weight: 600;
        }

        .flow-steps {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            align-items: center;
        }

        .flow-step {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
            position: relative;
        }

        .flow-step::after {
            content: '→';
            position: absolute;
            right: -25px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 1.5em;
            color: #667eea;
            font-weight: bold;
        }

        .flow-step:last-child::after {
            display: none;
        }

        .step-number {
            background: rgba(255, 255, 255, 0.2);
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 10px;
            font-weight: bold;
        }

        .step-title {
            font-weight: 600;
            margin-bottom: 8px;
        }

        .step-description {
            font-size: 0.9em;
            opacity: 0.9;
            line-height: 1.4;
        }

        /* Technology Stack */
        .tech-stack {
            background: white;
            border-radius: 15px;
            padding: 30px;
            border: 2px solid #e9ecef;
            margin-top: 30px;
        }

        .tech-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }

        .tech-category {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e9ecef;
        }

        .tech-category h4 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.2em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 8px;
        }

        .tech-list {
            list-style: none;
        }

        .tech-list li {
            padding: 8px 0;
            color: #495057;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .tech-list li:last-child {
            border-bottom: none;
        }

        .tech-badge {
            background: #667eea;
            color: white;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 0.75em;
        }

        /* Key Features */
        .features-section {
            background: white;
            border-radius: 15px;
            padding: 30px;
            border: 2px solid #e9ecef;
            margin-top: 30px;
        }

        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }

        .feature-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e9ecef;
            text-align: center;
        }

        .feature-icon {
            font-size: 2.5em;
            margin-bottom: 15px;
            color: #667eea;
        }

        .feature-title {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 10px;
        }

        .feature-description {
            color: #6c757d;
            font-size: 0.9em;
            line-height: 1.4;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .header h1 { font-size: 2em; }
            .layer-components { grid-template-columns: 1fr; }
            .flow-steps { grid-template-columns: 1fr; }
            .flow-step::after { display: none; }
            .tech-grid { grid-template-columns: 1fr; }
            .features-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏗️ BDD Test Automation Framework</h1>
            <div class="subtitle">Complaint AI Chatbot | Playwright + pytest-bdd Architecture</div>
        </div>

        <div class="architecture-container">
            <!-- Main Architecture Diagram -->
            <div class="main-diagram">
                <div class="diagram-title">📊 Framework Architecture Layers</div>
                
                <div class="architecture-layers">
                    <!-- Layer 1: Presentation & Configuration -->
                    <div class="layer presentation">
                        <div class="layer-header">
                            <div class="layer-title">🎯 Presentation & Configuration Layer</div>
                            <div class="layer-badge">Entry Point</div>
                        </div>
                        <div class="layer-components">
                            <div class="component">
                                <div class="component-name">Test Runner</div>
                                <div class="component-tech">test_runner.py</div>
                                <div class="component-description">CLI interface, environment selection, parallel execution</div>
                            </div>
                            <div class="component">
                                <div class="component-name">Configuration</div>
                                <div class="component-tech">Pydantic Settings</div>
                                <div class="component-description">Multi-environment configs, certificate processing</div>
                            </div>
                            <div class="component">
                                <div class="component-name">Environment Setup</div>
                                <div class="component-tech">.env files</div>
                                <div class="component-description">dev, qa, staging, production settings</div>
                            </div>
                        </div>
                        <div class="flow-arrow">⬇️</div>
                    </div>

                    <!-- Layer 2: Test Execution Engine -->
                    <div class="layer test-execution">
                        <div class="layer-header">
                            <div class="layer-title">🚀 Test Execution Engine</div>
                            <div class="layer-badge">BDD Core</div>
                        </div>
                        <div class="layer-components">
                            <div class="component">
                                <div class="component-name">BDD Features</div>
                                <div class="component-tech">Gherkin (.feature)</div>
                                <div class="component-description">complaint_capture.feature, dynamic_steps.feature</div>
                            </div>
                            <div class="component">
                                <div class="component-name">Step Definitions</div>
                                <div class="component-tech">pytest-bdd</div>
                                <div class="component-description">Given/When/Then step implementations</div>
                            </div>
                            <div class="component">
                                <div class="component-name">Test Context</div>
                                <div class="component-tech">pytest fixtures</div>
                                <div class="component-description">Shared test state, correlation tracking</div>
                            </div>
                            <div class="component">
                                <div class="component-name">Dynamic Execution</div>
                                <div class="component-tech">CSV-driven</div>
                                <div class="component-description">Conditional step execution based on data</div>
                            </div>
                        </div>
                        <div class="flow-arrow">⬇️</div>
                    </div>

                    <!-- Layer 3: API Client Layer -->
                    <div class="layer api-client">
                        <div class="layer-header">
                            <div class="layer-title">🔧 API Client Layer</div>
                            <div class="layer-badge">Playwright API</div>
                        </div>
                        <div class="layer-components">
                            <div class="component">
                                <div class="component-name">API Client</div>
                                <div class="component-tech">Playwright APIRequestContext</div>
                                <div class="component-description">High-performance API testing, no browser overhead</div>
                            </div>
                            <div class="component">
                                <div class="component-name">Certificate Auth</div>
                                <div class="component-tech">PFX → PEM conversion</div>
                                <div class="component-description">Secure mTLS authentication</div>
                            </div>
                            <div class="component">
                                <div class="component-name">Request Tracking</div>
                                <div class="component-tech">Correlation IDs</div>
                                <div class="component-description">Full request/response logging</div>
                            </div>
                            <div class="component">
                                <div class="component-name">Smart Retries</div>
                                <div class="component-tech">Exponential backoff</div>
                                <div class="component-description">Configurable retry strategies</div>
                            </div>
                            <div class="component">
                                <div class="component-name">Error Injection</div>
                                <div class="component-tech">Negative testing</div>
                                <div class="component-description">20+ error scenarios for comprehensive testing</div>
                            </div>
                        </div>
                        <div class="flow-arrow">⬇️</div>
                    </div>

                    <!-- Layer 4: Data Management -->
                    <div class="layer data-management">
                        <div class="layer-header">
                            <div class="layer-title">📊 Data Management Layer</div>
                            <div class="layer-badge">Data-Driven</div>
                        </div>
                        <div class="layer-components">
                            <div class="component">
                                <div class="component-name">CSV Data Loader</div>
                                <div class="component-tech">Robust encoding</div>
                                <div class="component-description">UTF-8/BOM handling, validation</div>
                            </div>
                            <div class="component">
                                <div class="component-name">Test Data</div>
                                <div class="component-tech">complaint_data.csv</div>
                                <div class="component-description">Test cases, workflow data, expected results</div>
                            </div>
                            <div class="component">
                                <div class="component-name">Request Templates</div>
                                <div class="component-tech">JSON templates</div>
                                <div class="component-description">API request patterns, error scenarios</div>
                            </div>
                            <div class="component">
                                <div class="component-name">Data Validation</div>
                                <div class="component-tech">Pydantic models</div>
                                <div class="component-description">Schema validation, type checking</div>
                            </div>
                        </div>
                        <div class="flow-arrow">⬇️</div>
                    </div>

                    <!-- Layer 5: Infrastructure -->
                    <div class="layer infrastructure">
                        <div class="layer-header">
                            <div class="layer-title">🗄️ Infrastructure Layer</div>
                            <div class="layer-badge">Persistence</div>
                        </div>
                        <div class="layer-components">
                            <div class="component">
                                <div class="component-name">Database Manager</div>
                                <div class="component-tech">Oracle + SQLAlchemy</div>
                                <div class="component-description">Connection pooling, transaction management</div>
                            </div>
                            <div class="component">
                                <div class="component-name">DB Utilities</div>
                                <div class="component-tech">Query helpers</div>
                                <div class="component-description">Chat history, complaint verification</div>
                            </div>
                            <div class="component">
                                <div class="component-name">Target APIs</div>
                                <div class="component-tech">Complaint AI Service</div>
                                <div class="component-description">Chatbot API, LLM service integration</div>
                            </div>
                            <div class="component">
                                <div class="component-name">External Services</div>
                                <div class="component-tech">HR/Customer lookup</div>
                                <div class="component-description">Integration validation</div>
                            </div>
                        </div>
                        <div class="flow-arrow">⬇️</div>
                    </div>

                    <!-- Layer 6: Reporting & Analytics -->
                    <div class="layer reporting">
                        <div class="layer-header">
                            <div class="layer-title">📈 Reporting & Analytics Layer</div>
                            <div class="layer-badge">Insights</div>
                        </div>
                        <div class="layer-components">
                            <div class="component">
                                <div class="component-name">HTML Reports</div>
                                <div class="component-tech">Rich dashboards</div>
                                <div class="component-description">Executive summary, step-level details</div>
                            </div>
                            <div class="component">
                                <div class="component-name">API Tracking</div>
                                <div class="component-tech">Request/response logs</div>
                                <div class="component-description">Performance metrics, correlation tracing</div>
                            </div>
                            <div class="component">
                                <div class="component-name">Logging System</div>
                                <div class="component-tech">Structured logging</div>
                                <div class="component-description">Debug, info, error levels with rotation</div>
                            </div>
                            <div class="component">
                                <div class="component-name">Metrics & KPIs</div>
                                <div class="component-tech">Test analytics</div>
                                <div class="component-description">Pass rates, performance trends</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Data Flow Diagram -->
            <div class="data-flow">
                <div class="flow-title">🔄 Test Execution Data Flow</div>
                <div class="flow-steps">
                    <div class="flow-step">
                        <div class="step-number">1</div>
                        <div class="step-title">Environment Setup</div>
                        <div class="step-description">Load configs, process certificates, initialize API client</div>
                    </div>
                    <div class="flow-step">
                        <div class="step-number">2</div>
                        <div class="step-title">Data Loading</div>
                        <div class="step-description">Parse CSV test data, validate fields, generate test cases</div>
                    </div>
                    <div class="flow-step">
                        <div class="step-number">3</div>
                        <div class="step-title">Test Generation</div>
                        <div class="step-description">Dynamic parametrization, BDD scenario creation</div>
                    </div>
                    <div class="flow-step">
                        <div class="step-number">4</div>
                        <div class="step-title">API Execution</div>
                        <div class="step-description">Initiate chat, execute workflow steps, track responses</div>
                    </div>
                    <div class="flow-step">
                        <div class="step-number">5</div>
                        <div class="step-title">Validation</div>
                        <div class="step-description">Verify API responses, check database persistence</div>
                    </div>
                    <div class="flow-step">
                        <div class="step-number">6</div>
                        <div class="step-title">Reporting</div>
                        <div class="step-description">Generate HTML reports, log results, cleanup</div>
                    </div>
                </div>
            </div>

            <!-- Technology Stack -->
            <div class="tech-stack">
                <div class="flow-title">🛠️ Technology Stack</div>
                <div class="tech-grid">
                    <div class="tech-category">
                        <h4>🧪 Testing Framework</h4>
                        <ul class="tech-list">
                            <li>pytest-bdd <span class="tech-badge">Core</span></li>
                            <li>Playwright APIRequestContext <span class="tech-badge">API</span></li>
                            <li>Gherkin <span class="tech-badge">BDD</span></li>
                            <li>pytest fixtures <span class="tech-badge">DI</span></li>
                        </ul>
                    </div>
                    <div class="tech-category">
                        <h4>⚙️ Configuration & Data</h4>
                        <ul class="tech-list">
                            <li>Pydantic Settings <span class="tech-badge">Config</span></li>
                            <li>CSV/JSON loaders <span class="tech-badge">Data</span></li>
                            <li>Environment configs <span class="tech-badge">Env</span></li>
                            <li>Certificate processing <span class="tech-badge">Auth</span></li>
                        </ul>
                    </div>
                    <div class="tech-category">
                        <h4>🗄️ Database & Persistence</h4>
                        <ul class="tech-list">
                            <li>Oracle Database <span class="tech-badge">DB</span></li>
                            <li>SQLAlchemy <span class="tech-badge">ORM</span></li>
                            <li>Connection pooling <span class="tech-badge">Pool</span></li>
                            <li>Transaction management <span class="tech-badge">TX</span></li>
                        </ul>
                    </div>
                    <div class="tech-category">
                        <h4>📊 Reporting & Analytics</h4>
                        <ul class="tech-list">
                            <li>HTML reports <span class="tech-badge">UI</span></li>
                            <li>JSON reporting <span class="tech-badge">API</span></li>
                            <li>Structured logging <span class="tech-badge">Logs</span></li>
                            <li>Performance metrics <span class="tech-badge">KPI</span></li>
                        </ul>
                    </div>
                </div>
            </div>

            <!-- Key Features -->
            <div class="features-section">
                <div class="flow-title">✨ Key Framework Features</div>
                <div class="features-grid">
                    <div class="feature-card">
                        <div class="feature-icon">🚀</div>
                        <div class="feature-title">High Performance</div>
                        <div class="feature-description">API-only testing with Playwright, no browser overhead, parallel execution support</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">📊</div>
                        <div class="feature-title">Data-Driven</div>
                        <div class="feature-description">CSV-based test data with dynamic step execution and comprehensive validation</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">🔧</div>
                        <div class="feature-title">Smart Error Testing</div>
                        <div class="feature-description">20+ built-in error injection scenarios for comprehensive negative testing</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">🔒</div>
                        <div class="feature-title">Enterprise Security</div>
                        <div class="feature-description">Certificate-based authentication, secure credential handling</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">📈</div>
                        <div class="feature-title">Rich Reporting</div>
                        <div class="feature-description">HTML dashboards with step-level details, API tracking, performance metrics</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">🌍</div>
                        <div class="feature-title">Multi-Environment</div>
                        <div class="feature-description">dev, qa, staging, production configs with environment-specific settings</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
