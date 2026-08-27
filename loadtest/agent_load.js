// k6 load test for the agent endpoint.
//
// LLM endpoints fail differently from ordinary APIs: they are slow, they are
// rate limited, and under concurrency you find 429s and timeouts long before you
// find CPU limits. That is what this measures.
//
//   k6 run loadtest/agent_load.js
//   k6 run -e BASE_URL=http://localhost:8000 -e VUS=20 loadtest/agent_load.js
//
// Requires a thin HTTP wrapper in front of harness.agent.run_agent (see
// README -> Load testing). Thresholds below are the pass/fail gate: k6 exits
// non-zero when one is breached, so this drops straight into CI.

import http from 'k6/http';
import { check } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const VUS = Number(__ENV.VUS || 10);

const rateLimited = new Rate('rate_limited');       // share of 429s
const emptyReplies = new Rate('empty_replies');     // 200 OK but nothing useful
const replyLatency = new Trend('reply_latency_ms'); // server-reported latency

export const options = {
  scenarios: {
    ramp: {
      executor: 'ramping-vus',
      startVUs: 1,
      stages: [
        { duration: '30s', target: VUS },      // ramp
        { duration: '2m', target: VUS },       // hold -- where rate limits appear
        { duration: '30s', target: 0 },        // ramp down
      ],
      gracefulRampDown: '30s',
    },
  },
  thresholds: {
    // Generous by web-API standards: a tool-calling turn is two model calls.
    http_req_duration: ['p(95)<8000', 'p(99)<15000'],
    http_req_failed: ['rate<0.02'],
    rate_limited: ['rate<0.05'],
    empty_replies: ['rate<0.01'],
  },
};

const MESSAGES = [
  { message: "Hey, where's my order ORD-1041?", account_id: 'ACC-77' },
  { message: 'السلام عليكم، وين وصل طلبي ORD-1041؟', account_id: 'ACC-77' },
  { message: 'My order still hasn’t arrived and I’m getting frustrated.', account_id: 'ACC-77' },
  { message: 'Any update on ORD-1044?', account_id: 'ACC-77' },
];

export default function () {
  const payload = MESSAGES[Math.floor(Math.random() * MESSAGES.length)];

  const res = http.post(`${BASE_URL}/reply`, JSON.stringify(payload), {
    headers: { 'Content-Type': 'application/json' },
    timeout: '60s',
  });

  rateLimited.add(res.status === 429);

  let body = {};
  try {
    body = res.json();
  } catch (e) {
    body = {};
  }

  emptyReplies.add(res.status === 200 && !(body.reply && body.reply.trim().length));
  if (body.latency_ms) replyLatency.add(body.latency_ms);

  check(res, {
    'status is 200': (r) => r.status === 200,
    'reply is not empty': () => Boolean(body.reply && body.reply.trim().length),
    // Under load the interesting failure is a truncated or degraded answer, not
    // a 500. Keep an eye on this one specifically.
    'reply is plausibly complete': () => Boolean(body.reply && body.reply.length > 20),
  });
}
