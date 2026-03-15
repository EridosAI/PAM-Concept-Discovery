// CloudFlare Worker — proxies PAM analysis & novel requests to vast.ai GPU server
// Deploy at e.g. https://pam-server.jasondury.workers.dev
//
// Setup:
//   1. Create a new worker in CF dashboard (or `npx wrangler init pam-server`)
//   2. Set the environment variable:
//        npx wrangler secret put PAM_SERVER_URL
//        → enter: http://141.0.85.200:42501
//   3. Deploy:
//        npx wrangler deploy worker-pam-server.js --name pam-server
//
// Then update pam_10k_demo.jsx:
//   const SERVER_URL = "https://pam-server.jasondury.workers.dev";

export default {
  async fetch(request, env) {
    const BACKEND = env.PAM_SERVER_URL || 'http://141.0.85.200:42501';

    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    // CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    const url = new URL(request.url);
    const path = url.pathname;

    // Allowed routes
    const allowedPaths = ['/health', '/api/analyze', '/api/rate-info'];
    const isNovelPath = path.startsWith('/novel/');
    if (!allowedPaths.includes(path) && !isNovelPath) {
      return new Response('Not found', { status: 404, headers: corsHeaders });
    }

    // Size limit for uploads (5MB)
    const contentLength = parseInt(request.headers.get('Content-Length') || '0');
    if (contentLength > 5 * 1024 * 1024) {
      return new Response(JSON.stringify({ error: 'File too large (max 5MB)' }), {
        status: 413,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    try {
      // Forward request to backend
      const backendUrl = BACKEND + path;
      const backendHeaders = new Headers();

      // Forward content-type for POST requests
      if (request.headers.get('Content-Type')) {
        backendHeaders.set('Content-Type', request.headers.get('Content-Type'));
      }

      const fetchOptions = {
        method: request.method,
        headers: backendHeaders,
      };

      // Forward body for POST
      if (request.method === 'POST') {
        fetchOptions.body = await request.arrayBuffer();
      }

      const backendResp = await fetch(backendUrl, fetchOptions);

      // For SSE responses (analyze endpoint), stream through
      const contentType = backendResp.headers.get('Content-Type') || '';

      if (contentType.includes('text/event-stream')) {
        // Stream SSE through the worker
        return new Response(backendResp.body, {
          status: backendResp.status,
          headers: {
            ...corsHeaders,
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
          },
        });
      }

      // For JSON responses, forward as-is
      const body = await backendResp.text();
      return new Response(body, {
        status: backendResp.status,
        headers: {
          ...corsHeaders,
          'Content-Type': contentType || 'application/json',
        },
      });

    } catch (err) {
      return new Response(JSON.stringify({
        error: 'GPU server unavailable. It may be starting up or temporarily offline.',
        detail: err.message,
      }), {
        status: 502,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }
  }
};
