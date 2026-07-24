"""FastAPI application for the ClawVault dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from claw_vault import __version__
from claw_vault.dashboard.api import router as api_router
from claw_vault.proxy.provider_adapter import router as provider_adapter_router

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(
        title="ClawVault Dashboard",
        version=__version__,
        description="Security monitoring dashboard for ClawVault",
    )

    app.include_router(api_router, prefix="/api")
    app.include_router(provider_adapter_router)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return index_file.read_text(encoding="utf-8")
        return _fallback_html()

    return app


def _fallback_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ClawVault Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: { extend: { colors: { vault: { 50:'#f0fdf4', 500:'#22c55e', 600:'#16a34a', 700:'#15803d', 900:'#14532d' }}}}
        }
    </script>
</head>
<body class="bg-gray-950 text-gray-100 min-h-screen">
    <div id="app" class="max-w-6xl mx-auto px-4 py-8">
        <!-- Header -->
        <header class="flex items-center justify-between mb-8">
            <div class="flex items-center gap-3">
                <div class="w-10 h-10 bg-vault-600 rounded-lg flex items-center justify-center text-xl">🛡️</div>
                <div>
                    <h1 class="text-2xl font-bold">ClawVault</h1>
                    <p class="text-gray-400 text-sm">Security Dashboard</p>
                </div>
            </div>
            <div id="status" class="flex items-center gap-2">
                <span class="w-2 h-2 bg-vault-500 rounded-full animate-pulse"></span>
                <span class="text-sm text-gray-400">Active</span>
            </div>
        </header>

        <!-- Stats Cards -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div class="bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div class="text-gray-400 text-sm mb-1">Interceptions</div>
                <div id="stat-interceptions" class="text-3xl font-bold text-vault-500">-</div>
            </div>
            <div class="bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div class="text-gray-400 text-sm mb-1">Tokens Used</div>
                <div id="stat-tokens" class="text-3xl font-bold text-blue-400">-</div>
            </div>
            <div class="bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div class="text-gray-400 text-sm mb-1">Cost (Today)</div>
                <div id="stat-cost" class="text-3xl font-bold text-yellow-400">-</div>
            </div>
            <div class="bg-gray-900 rounded-xl p-5 border border-gray-800">
                <div class="text-gray-400 text-sm mb-1">Risk Score</div>
                <div id="stat-risk" class="text-3xl font-bold text-red-400">-</div>
            </div>
        </div>

        <!-- Budget Bar -->
        <div class="bg-gray-900 rounded-xl p-5 border border-gray-800 mb-8">
            <div class="flex justify-between items-center mb-2">
                <span class="text-gray-400 text-sm">Daily Token Budget</span>
                <span id="budget-text" class="text-sm text-gray-400">- / -</span>
            </div>
            <div class="w-full bg-gray-800 rounded-full h-3">
                <div id="budget-bar" class="bg-vault-500 h-3 rounded-full transition-all" style="width:0%"></div>
            </div>
        </div>

        <!-- Recent Events -->
        <div class="bg-gray-900 rounded-xl border border-gray-800">
            <div class="px-5 py-4 border-b border-gray-800 flex justify-between items-center">
                <h2 class="font-semibold">Recent Events</h2>
                <button onclick="refreshData()" class="text-sm text-vault-500 hover:text-vault-400">Refresh</button>
            </div>
            <div id="events-list" class="divide-y divide-gray-800">
                <div class="px-5 py-8 text-center text-gray-500">Loading events...</div>
            </div>
        </div>
    </div>

    <script>
    const RISK_COLORS = { safe:'text-gray-400', low:'text-blue-400', medium:'text-yellow-400', high:'text-orange-400', critical:'text-red-400' };
    const ACTION_ICONS = { allow:'✅', block:'🚨', sanitize:'🛡️', ask_user:'⚠️' };

    async function fetchAPI(path) {
        try { const r = await fetch('/api' + path); return await r.json(); }
        catch(e) { console.error(e); return null; }
    }

    async function refreshData() {
        const [summary, budget, events] = await Promise.all([
            fetchAPI('/summary'), fetchAPI('/budget'), fetchAPI('/events?limit=20')
        ]);

        if (summary) {
            document.getElementById('stat-interceptions').textContent = summary.interceptions || 0;
            document.getElementById('stat-tokens').textContent = (summary.total_tokens || 0).toLocaleString();
            document.getElementById('stat-cost').textContent = '$' + (summary.total_cost_usd || 0).toFixed(2);
            document.getElementById('stat-risk').textContent = (summary.max_risk_score || 0).toFixed(1);
        }

        if (budget) {
            const pct = Math.min(budget.daily_pct || 0, 100);
            document.getElementById('budget-bar').style.width = pct + '%';
            document.getElementById('budget-bar').className =
                'h-3 rounded-full transition-all ' + (pct > 80 ? 'bg-red-500' : pct > 60 ? 'bg-yellow-500' : 'bg-vault-500');
            document.getElementById('budget-text').textContent =
                (budget.daily_used||0).toLocaleString() + ' / ' + (budget.daily_limit||0).toLocaleString();
        }

        if (events && events.length) {
            const el = document.getElementById('events-list');
            el.innerHTML = events.map(e => `
                <div class="px-5 py-3 flex items-center gap-3 hover:bg-gray-800/50">
                    <span class="text-lg">${ACTION_ICONS[e.action_taken] || '📝'}</span>
                    <div class="flex-1 min-w-0">
                        <div class="text-sm truncate">${e.api_endpoint || 'N/A'}</div>
                        <div class="text-xs text-gray-500">${e.details || e.action_taken}</div>
                    </div>
                    <span class="text-xs ${RISK_COLORS[e.risk_level] || 'text-gray-400'}">${e.risk_level}</span>
                    <span class="text-xs text-gray-500">${new Date(e.timestamp).toLocaleTimeString()}</span>
                </div>
            `).join('');
        }
    }

    refreshData();
    setInterval(refreshData, 5000);
    </script>
</body>
</html>"""
