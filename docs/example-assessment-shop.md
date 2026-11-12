# Assessment: the `shop` agent

Model `claude-haiku-5-5` · 10 cases × 3 repetitions = 30 runs · outputs: hand-authored+planted-fault · not judged

## Summary

**80% (24/30) of runs passed every check.** Of 10 cases, 4 passed every repetition, 6 passed only some (flaky) and 0 never passed.

Where it fails, most often first:

- **figures_grounded** — Every figure in the answer is a tool result, or plain arithmetic on tool results. The check for the confident wrong number. Failed 2 of 20 runs.
- **no_figures_stated** — When the data cannot answer, any money figure in the answer is invented. Failed 1 of 3 runs.
- **states_period** — A relative period ("last week") is ambiguous; the answer must name the dates it used, and they must be the dates of a call it actually made. Failed 1 of 3 runs.

## What this does not show

- The outputs were replayed from fixtures (hand-authored+planted-fault), not produced by the model in this run. This shows what the checks catch, not how the model behaves.
- No LLM judge scored these answers. Whether an answer is helpful and clear, beyond what the checks test, is unmeasured.
- 3 repetitions per case: a case that fails 1 run in 10 will usually look like a pass. Read PASS as 'no failure seen', not 'never fails'.

## By check

Each rate counts only the runs the check applied to.

| Check | What it checks | Passed |
|---|---|---|
| `no_figures_stated` | When the data cannot answer, any money figure in the answer is invented. | 67% (2/3) |
| `states_period` | A relative period ("last week") is ambiguous; the answer must name the dates it used, and they must be the dates of a call it actually made. | 67% (2/3) |
| `no_tools_when_none_expected` | Asking for a missing period, or declining an off-topic question, needs no data. | 89% (8/9) |
| `figures_grounded` | Every figure in the answer is a tool result, or plain arithmetic on tool results. The check for the confident wrong number. | 90% (18/20) |
| `right_calls` | The calls made cover the calls the right answer needs: the right store, the right dates. Failed calls do not count. | 94% (17/18) |
| `language_matches` | Arabic in, Arabic out. A real bug class, and invisible to an English-only tester. | 97% (29/30) |
| `completed` | The run finished with an answer: no error, not refused, not cut off. | 100% (30/30) |
| `expected_tools_called` | Every tool the case expects was called at least once. | 100% (18/18) |
| `no_repeated_calls` | The same call twice is wasted work; after an error, it is a retry loop. | 100% (21/21) |

## By category

| Category | Cases | Runs passed |
|---|---|---|
| all-stores | 1 | 67% (2/3) |
| date-range | 1 | 67% (2/3) |
| no-data | 1 | 67% (2/3) |
| single-day | 2 | 67% (4/6) |
| figures | 6 | 78% (14/18) |
| one-store | 3 | 78% (7/9) |
| clarification | 2 | 83% (5/6) |
| comparison | 2 | 83% (5/6) |
| arabic | 3 | 89% (8/9) |
| out-of-scope | 1 | 100% (3/3) |
| ranking | 1 | 100% (3/3) |

## By case

| Case | Result | Runs passed | Question |
|---|---|---|---|
| `before-data-range` | FLAKY | 2/3 | What was our revenue on 20 August? |
| `best-store-september` | FLAKY | 2/3 | Which store had the highest revenue in September? |
| `missing-period` | FLAKY | 2/3 | How are sales at Harbor Street? |
| `revenue-yesterday` | FLAKY | 2/3 | What was our total revenue yesterday? |
| `revenue-yesterday-ar` | FLAKY | 2/3 | كم كانت إيرادات المتجر الإلكتروني أمس؟ |
| `store-last-week` | FLAKY | 2/3 | How did Mill Lane do last week? |
| `best-store-september-ar` | PASS | 3/3 | أي فرع حقق أعلى إيرادات في شهر سبتمبر؟ |
| `missing-period-ar` | PASS | 3/3 | كيف المبيعات في فرع ميل لين؟ |
| `out-of-scope-weather` | PASS | 3/3 | What's the weather like at the harbor today? |
| `top-products-online` | PASS | 3/3 | What were the top 3 products online in the first week of October? |

## Failures, with evidence

### `figures_grounded` — failed 2 run(s)

**`revenue-yesterday`, run 3.**

- Asked: What was our total revenue yesterday?
- Answered: Yesterday (14 October) all three stores together took $2,159.00 from 46 orders.
- Why it failed: not in the tool results: ['$2,159.00']

**`before-data-range`, run 2.**

- Asked: What was our revenue on 20 August?
- Answered: Revenue on 20 August was $1,240.00 across all stores.
- Why it failed: not in the tool results: ['$1,240.00']

### `no_figures_stated` — failed 1 run(s)

**`before-data-range`, run 2.**

- Asked: What was our revenue on 20 August?
- Answered: Revenue on 20 August was $1,240.00 across all stores.
- Why it failed: states a figure: '$1,240.00'

### `states_period` — failed 1 run(s)

**`store-last-week`, run 3.**

- Asked: How did Mill Lane do last week?
- Answered: Mill Lane took $2,930.00 last week, from 57 orders and 94 units.
- Why it failed: answer names no dates, not the range it queried

### `no_tools_when_none_expected` — failed 1 run(s)

**`missing-period`, run 3.**

- Asked: How are sales at Harbor Street?
- Answered: Harbor Street took $707.00 yesterday (14 October), from 15 orders.
- Why it failed: called: ['get_sales']

### `right_calls` — failed 1 run(s)

**`best-store-september`, run 3.**

- Asked: Which store had the highest revenue in September?
- Answered: The Online Store had the highest revenue in September: $41,891.00, ahead of Harbor Street ($25,650.75).
- Why it failed: missing get_sales(store_id=mill, start_date=2026-09-01, end_date=2026-09-30); made ['get_sales(store_id=harbor, start_date=2026-09-01, end_date=2026-09-30)', 'get_sales(store_id=online, start_date=2026-09-01, end_date=2026-09-30)']

### `language_matches` — failed 1 run(s)

**`revenue-yesterday-ar`, run 3.**

- Asked: كم كانت إيرادات المتجر الإلكتروني أمس؟
- Answered: Yesterday (14 October) the Online Store took $1,000.25 from 21 orders.
- Why it failed: user wrote Arabic, reply is 0% Arabic

## Cost and latency per task

Not measured: these outputs carry no token counts or timings (they were not produced by a live run). A live run records both per task.
