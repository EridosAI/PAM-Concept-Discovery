// CloudFlare Worker — proxies explain requests to Anthropic
// Deploy at e.g. https://pam-explain.eridos.workers.dev
//
// Setup:
//   npx wrangler secret put ANTHROPIC_API_KEY
//   npx wrangler deploy worker.js

export default {
  async fetch(request, env) {
    // CORS preflight
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',  // TODO: lock to 'https://eridos.ai' in production
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    // Only accept POST
    if (request.method !== 'POST') {
      return new Response('Method not allowed', { status: 405, headers: corsHeaders });
    }

    // Parse structured input (no arbitrary prompts)
    let data;
    try {
      data = await request.json();
    } catch {
      return new Response('Invalid JSON', { status: 400, headers: corsHeaders });
    }

    const required = ['chunkText', 'clusterLabel', 'novelTitle', 'k'];
    if (!required.every(f => data[f])) {
      return new Response('Missing fields', { status: 400, headers: corsHeaders });
    }
    if (data.chunkText.length > 2000) {
      return new Response('Input too long', { status: 400, headers: corsHeaders });
    }

    // Build prompt server-side (client never controls system prompt)
    const systemPrompt = `You are an expert literary analyst examining clusters discovered by an unsupervised AI model (Predictive Associative Memory) trained on 10,000 Project Gutenberg novels. The model groups text chunks by temporal co-occurrence patterns — passages that serve similar narrative STRUCTURAL functions tend to appear in similar sequential contexts across different novels, regardless of their surface content.

Important: the model knows nothing about themes, topics, or meaning. It only knows what kinds of passages tend to appear before and after each other. Two passages can be structurally identical (same rhetorical pattern, same pacing, same position in a narrative arc) while being about completely different things.

When the cluster label seems wrong for the chunk, this is often the most interesting case. Look for the STRUCTURAL parallel — the rhetorical pattern, pacing, narrative position, or formal technique that connects them despite different surface content. Explain what structural feature the model likely detected. For example: a passage about maritime ambition might share the rhetorical structure of romantic declaration (earnest appeal, worthiness argument, life-changing choice). Name the structural parallel explicitly.

Explain in 2-3 sentences:
1. What narrative/structural function this chunk serves
2. Why it belongs in this cluster — what structural feature connects it to the other samples, even if the surface content differs
3. What's interesting or surprising about the grouping

Be specific about the text. Reference actual phrases. Be concise.`;

    // Build user message from structured fields
    const sampleLines = (data.sampleTexts || [])
      .map(s => `- From "${s.book}": ${s.text}`)
      .join('\n');

    const userMsg = `This chunk from "${data.novelTitle}" by ${data.author || 'unknown'} was assigned to cluster "${data.clusterLabel}" (k=${data.k}, ${data.clusterSize || '?'} chunks across ${data.nBooks || '?'} books).${data.clusterDescription ? '\n\nDescription: ' + data.clusterDescription : ''}

THE CHUNK:
${data.chunkText}

${data.prevChunkText ? 'CONTEXT (previous chunk):\n' + data.prevChunkText + '\n' : ''}${data.nextChunkText ? 'CONTEXT (next chunk):\n' + data.nextChunkText + '\n' : ''}${sampleLines ? 'OTHER BOOKS IN THIS CLUSTER (samples):\n' + sampleLines + '\n' : ''}
Why does this chunk belong in this cluster? What's interesting about this grouping?`;

    // Call Anthropic
    try {
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': env.ANTHROPIC_API_KEY,
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify({
          model: 'claude-sonnet-4-6',
          max_tokens: 300,
          system: systemPrompt,
          messages: [{ role: 'user', content: userMsg }],
        }),
      });

      if (!response.ok) {
        const errText = await response.text().catch(() => '');
        return new Response(JSON.stringify({ error: `Anthropic API error: ${response.status}` }), {
          status: 502,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        });
      }

      const result = await response.json();
      const text = result.content?.[0]?.text || 'Unable to generate explanation.';

      return new Response(JSON.stringify({ explanation: text }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    } catch (err) {
      return new Response(JSON.stringify({ error: 'Worker error: ' + err.message }), {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }
  }
};
