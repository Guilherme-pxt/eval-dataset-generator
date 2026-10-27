# Eval run report

## Dataset

- Log lines read: 303, records kept: 300, rejected: 3, PII spans redacted: 7
- Unique prompts after near-duplicate collapse: 133; clusters: 27; selected for review: 48 {'representative': 27, 'failure_signal': 29, 'novel_in_cluster': 7, 'rare_cluster': 3}
- Against the synthetic ground truth: cluster purity 0.72, intents covered 21/24 (rare: 6/8), production failure rate 16.3% in the log vs 58.3% in the selection
- Review (oracle): accepted 30, edited 16, rejected 2 -> 46 eval items; draft acceptance rate 0.652

## Overall

| version | model | prompt | items | valid JSON | action acc. | det score | det pass | judge (1-5) | combined | judge/det agreement | mean latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | qwen2.5:7b | v1 | 46 | 1.000 | 0.957 | 0.924 | 0.913 | 4.59 | 0.910 | 0.935 | 5.36 |
| candidate | qwen2.5-coder:7b | v1 | 46 | 1.000 | 0.957 | 0.935 | 0.935 | 4.65 | 0.924 | 0.913 | 4.37 |
| prompt_change | qwen2.5:7b | v1-no-policy | 46 | 1.000 | 1.000 | 0.967 | 0.935 | 4.63 | 0.938 | 0.804 | 5.26 |

### By expected action

| version | answer (n) det / combined | tool_call (n) det / combined | decline (n) det / combined |
|---|---|---|---|
| baseline | (34) 0.897 / 0.901 | (7) 1.000 / 0.946 | (5) 1.000 / 0.925 |
| candidate | (34) 0.912 / 0.926 | (7) 1.000 / 0.857 | (5) 1.000 / 1.000 |
| prompt_change | (34) 0.956 / 0.941 | (7) 1.000 / 0.929 | (5) 1.000 / 0.925 |

## Gate: candidate vs baseline: **FAIL**

- combined: 0.9103 -> 0.9239 (delta +0.014, paired bootstrap 95% CI [-0.068, +0.087])
- items newly failing: 4, items fixed: 4
- reason: cluster 'invoiceid / paid / client' (n=3) dropped by 0.167 (limit 0.15)
- reason: cluster 'fee / card / transaction' (n=3) dropped by 0.167 (limit 0.15)
- reason: 4 items newly failing (limit 3)

| cluster | n | baseline | candidate | delta | gated | regressed |
|---|---|---|---|---|---|---|
| automatically / interest / add | 1 | 1.000 | 0.000 | -1.000 | no |  |
| invoiceid / paid / client | 3 | 1.000 | 0.833 | -0.167 | yes | YES |
| fee / card / transaction | 3 | 1.000 | 0.833 | -0.167 | yes | YES |
| pay / includ / user | 1 | 1.000 | 0.875 | -0.125 | no |  |
| refund / accountid / wrong | 3 | 0.875 | 0.833 | -0.042 | yes |  |
| backup / full / workspac | 1 | 1.000 | 1.000 | +0.000 | no |  |
| trial / fre / card | 2 | 1.000 | 1.000 | +0.000 | no |  |
| day / trial / thx | 1 | 1.000 | 1.000 | +0.000 | no |  |
| lin / item / tax | 1 | 1.000 | 1.000 | +0.000 | no |  |
| starter / limit / plan | 1 | 1.000 | 1.000 | +0.000 | no |  |
| payer / up / automatic | 1 | 1.000 | 1.000 | +0.000 | no |  |
| support / sso / availabl | 1 | 1.000 | 1.000 | +0.000 | no |  |
| sso / plan / whcih | 1 | 1.000 | 1.000 | +0.000 | no |  |
| payment / charg / card | 2 | 1.000 | 1.000 | +0.000 | no |  |
| data / stored / download | 1 | 1.000 | 1.000 | +0.000 | no |  |
| lat / fee / maximum | 1 | 1.000 | 1.000 | +0.000 | no |  |
| export / invoic / yen | 1 | 1.000 | 1.000 | +0.000 | no |  |
| payout / tim / bank | 1 | 1.000 | 1.000 | +0.000 | no |  |
| cup / won / world | 1 | 1.000 | 1.000 | +0.000 | no |  |
| email / out / overdu | 1 | 1.000 | 1.000 | +0.000 | no |  |
| sav / year / full | 1 | 0.875 | 0.875 | +0.000 | no |  |
| help / received / chargeback | 1 | 1.000 | 1.000 | +0.000 | no |  |
| growth / pric / plan | 3 | 0.917 | 1.000 | +0.083 | yes |  |
| bill / offer / discount | 9 | 0.903 | 1.000 | +0.097 | yes |  |
| month / growth / access | 2 | 0.500 | 0.625 | +0.125 | no |  |
| annual / plan / discount | 1 | 0.500 | 1.000 | +0.500 | no |  |
| customer / weekend / open | 1 | 0.000 | 1.000 | +1.000 | no |  |

## Gate: prompt_change vs baseline: **FAIL**

- combined: 0.9103 -> 0.9375 (delta +0.027, paired bootstrap 95% CI [-0.025, +0.090])
- items newly failing: 4, items fixed: 1
- reason: 4 items newly failing (limit 3)

| cluster | n | baseline | candidate | delta | gated | regressed |
|---|---|---|---|---|---|---|
| backup / full / workspac | 1 | 1.000 | 0.750 | -0.250 | no |  |
| email / out / overdu | 1 | 1.000 | 0.750 | -0.250 | no |  |
| pay / includ / user | 1 | 1.000 | 0.875 | -0.125 | no |  |
| cup / won / world | 1 | 1.000 | 0.875 | -0.125 | no |  |
| sav / year / full | 1 | 0.875 | 0.750 | -0.125 | no |  |
| refund / accountid / wrong | 3 | 0.875 | 0.833 | -0.042 | yes |  |
| growth / pric / plan | 3 | 0.917 | 0.917 | +0.000 | yes |  |
| trial / fre / card | 2 | 1.000 | 1.000 | +0.000 | no |  |
| day / trial / thx | 1 | 1.000 | 1.000 | +0.000 | no |  |
| lin / item / tax | 1 | 1.000 | 1.000 | +0.000 | no |  |
| starter / limit / plan | 1 | 1.000 | 1.000 | +0.000 | no |  |
| payer / up / automatic | 1 | 1.000 | 1.000 | +0.000 | no |  |
| support / sso / availabl | 1 | 1.000 | 1.000 | +0.000 | no |  |
| sso / plan / whcih | 1 | 1.000 | 1.000 | +0.000 | no |  |
| payment / charg / card | 2 | 1.000 | 1.000 | +0.000 | no |  |
| data / stored / download | 1 | 1.000 | 1.000 | +0.000 | no |  |
| lat / fee / maximum | 1 | 1.000 | 1.000 | +0.000 | no |  |
| export / invoic / yen | 1 | 1.000 | 1.000 | +0.000 | no |  |
| invoiceid / paid / client | 3 | 1.000 | 1.000 | +0.000 | yes |  |
| payout / tim / bank | 1 | 1.000 | 1.000 | +0.000 | no |  |
| automatically / interest / add | 1 | 1.000 | 1.000 | +0.000 | no |  |
| help / received / chargeback | 1 | 1.000 | 1.000 | +0.000 | no |  |
| fee / card / transaction | 3 | 1.000 | 1.000 | +0.000 | yes |  |
| bill / offer / discount | 9 | 0.903 | 0.944 | +0.042 | yes |  |
| annual / plan / discount | 1 | 0.500 | 0.625 | +0.125 | no |  |
| month / growth / access | 2 | 0.500 | 0.875 | +0.375 | no |  |
| customer / weekend / open | 1 | 0.000 | 1.000 | +1.000 | no |  |

## Timing

| step | wall time (s) |
|---|---|
| ingest | 0.0 |
| cluster | 0.25 |
| draft | 716.4 |
| review | 0.01 |
| score: baseline predict | 246.9 |
| score: baseline judge | 345.5 |
| score: candidate predict | 201.1 |
| score: candidate judge | 354.0 |
| score: prompt_change predict | 242.2 |
| score: prompt_change judge | 233.3 |
| run_all | 2339.8 |
