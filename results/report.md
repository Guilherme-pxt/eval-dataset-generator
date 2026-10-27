# Eval run report

## Dataset

- Log lines read: 303, records kept: 300, rejected: 3, PII spans redacted: 7
- Unique prompts after near-duplicate collapse: 133; clusters: 27; selected for review: 48 {'representative': 27, 'failure_signal': 29, 'novel_in_cluster': 7, 'rare_cluster': 3}
- Against the synthetic ground truth: cluster purity 0.72, intents covered 21/24 (rare: 6/8), production failure rate 16.3% in the log vs 58.3% in the selection
- Review (oracle): accepted 31, edited 15, rejected 2 -> 46 eval items; draft acceptance rate 0.674

## Overall

| version | model | prompt | items | valid JSON | action acc. | det score | det pass | judge (1-5) | combined | judge/det agreement | mean latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | qwen2.5:7b | v1 | 46 | 1.000 | 0.957 | 0.957 | 0.957 | 4.59 | 0.927 | 0.935 | 4.51 |
| candidate | qwen2.5-coder:7b | v1 | 46 | 1.000 | 0.957 | 0.957 | 0.957 | 4.65 | 0.935 | 0.935 | 4.54 |
| prompt_change | qwen2.5:7b | v1-no-policy | 46 | 1.000 | 1.000 | 0.978 | 0.957 | 4.63 | 0.943 | 0.826 | 5.26 |
| retrieval_outage | qwen2.5:7b | v1-no-docs | 46 | 1.000 | 0.609 | 0.371 | 0.348 | 2.50 | 0.373 | 0.870 | 4.43 |

### By expected action

| version | answer (n) det / combined | tool_call (n) det / combined | decline (n) det / combined |
|---|---|---|---|
| baseline | (34) 0.941 / 0.923 | (7) 1.000 / 0.946 | (5) 1.000 / 0.925 |
| candidate | (34) 0.941 / 0.941 | (7) 1.000 / 0.857 | (5) 1.000 / 1.000 |
| prompt_change | (34) 0.971 / 0.949 | (7) 1.000 / 0.929 | (5) 1.000 / 0.925 |
| retrieval_outage | (34) 0.149 / 0.178 | (7) 1.000 / 0.911 | (5) 1.000 / 0.950 |

## Gate: candidate vs baseline: **FAIL**

- thresholds: overall drop > 0.05, cluster drop > 0.15 (clusters with n >= 3), newly failing items > 3
- combined: 0.9266 -> 0.9348 (delta +0.008, paired bootstrap 95% CI [-0.065, +0.079])
- items newly failing: 3, items fixed: 3
- reason: cluster 'invoiceid / paid / client' (n=3) dropped by 0.167 (limit 0.15)

| cluster | n | baseline | candidate | delta | gated | regressed |
|---|---|---|---|---|---|---|
| automatically / interest / add | 1 | 1.000 | 0.000 | -1.000 | no |  |
| invoiceid / paid / client | 3 | 1.000 | 0.833 | -0.167 | yes | YES |
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
| annual / plan / discount | 1 | 1.000 | 1.000 | +0.000 | no |  |
| lat / fee / maximum | 1 | 1.000 | 1.000 | +0.000 | no |  |
| export / invoic / yen | 1 | 1.000 | 1.000 | +0.000 | no |  |
| payout / tim / bank | 1 | 1.000 | 1.000 | +0.000 | no |  |
| cup / won / world | 1 | 1.000 | 1.000 | +0.000 | no |  |
| email / out / overdu | 1 | 1.000 | 1.000 | +0.000 | no |  |
| sav / year / full | 1 | 0.875 | 0.875 | +0.000 | no |  |
| help / received / chargeback | 1 | 1.000 | 1.000 | +0.000 | no |  |
| fee / card / transaction | 3 | 1.000 | 1.000 | +0.000 | yes |  |
| bill / offer / discount | 9 | 0.931 | 1.000 | +0.069 | yes |  |
| growth / pric / plan | 3 | 0.917 | 1.000 | +0.083 | yes |  |
| month / growth / access | 2 | 0.500 | 0.625 | +0.125 | no |  |
| customer / weekend / open | 1 | 0.000 | 1.000 | +1.000 | no |  |

## Gate: prompt_change vs baseline: **FAIL**

- thresholds: overall drop > 0.05, cluster drop > 0.15 (clusters with n >= 3), newly failing items > 3
- combined: 0.9266 -> 0.9429 (delta +0.016, paired bootstrap 95% CI [-0.038, +0.082])
- items newly failing: 5, items fixed: 2
- reason: 5 items newly failing (limit 3)

| cluster | n | baseline | candidate | delta | gated | regressed |
|---|---|---|---|---|---|---|
| annual / plan / discount | 1 | 1.000 | 0.625 | -0.375 | no |  |
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
| bill / offer / discount | 9 | 0.931 | 0.972 | +0.042 | yes |  |
| month / growth / access | 2 | 0.500 | 0.875 | +0.375 | no |  |
| customer / weekend / open | 1 | 0.000 | 1.000 | +1.000 | no |  |

## Gate: candidate vs baseline: **PASS**

- thresholds: overall drop > 0.05, cluster drop > 0.15 (clusters with n >= 5), newly failing items > 5
- combined: 0.9266 -> 0.9348 (delta +0.008, paired bootstrap 95% CI [-0.065, +0.079])
- items newly failing: 3, items fixed: 3

| cluster | n | baseline | candidate | delta | gated | regressed |
|---|---|---|---|---|---|---|
| automatically / interest / add | 1 | 1.000 | 0.000 | -1.000 | no |  |
| invoiceid / paid / client | 3 | 1.000 | 0.833 | -0.167 | no |  |
| pay / includ / user | 1 | 1.000 | 0.875 | -0.125 | no |  |
| refund / accountid / wrong | 3 | 0.875 | 0.833 | -0.042 | no |  |
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
| annual / plan / discount | 1 | 1.000 | 1.000 | +0.000 | no |  |
| lat / fee / maximum | 1 | 1.000 | 1.000 | +0.000 | no |  |
| export / invoic / yen | 1 | 1.000 | 1.000 | +0.000 | no |  |
| payout / tim / bank | 1 | 1.000 | 1.000 | +0.000 | no |  |
| cup / won / world | 1 | 1.000 | 1.000 | +0.000 | no |  |
| email / out / overdu | 1 | 1.000 | 1.000 | +0.000 | no |  |
| sav / year / full | 1 | 0.875 | 0.875 | +0.000 | no |  |
| help / received / chargeback | 1 | 1.000 | 1.000 | +0.000 | no |  |
| fee / card / transaction | 3 | 1.000 | 1.000 | +0.000 | no |  |
| bill / offer / discount | 9 | 0.931 | 1.000 | +0.069 | yes |  |
| growth / pric / plan | 3 | 0.917 | 1.000 | +0.083 | no |  |
| month / growth / access | 2 | 0.500 | 0.625 | +0.125 | no |  |
| customer / weekend / open | 1 | 0.000 | 1.000 | +1.000 | no |  |

## Gate: prompt_change vs baseline: **PASS**

- thresholds: overall drop > 0.05, cluster drop > 0.15 (clusters with n >= 5), newly failing items > 5
- combined: 0.9266 -> 0.9429 (delta +0.016, paired bootstrap 95% CI [-0.038, +0.082])
- items newly failing: 5, items fixed: 2

| cluster | n | baseline | candidate | delta | gated | regressed |
|---|---|---|---|---|---|---|
| annual / plan / discount | 1 | 1.000 | 0.625 | -0.375 | no |  |
| backup / full / workspac | 1 | 1.000 | 0.750 | -0.250 | no |  |
| email / out / overdu | 1 | 1.000 | 0.750 | -0.250 | no |  |
| pay / includ / user | 1 | 1.000 | 0.875 | -0.125 | no |  |
| cup / won / world | 1 | 1.000 | 0.875 | -0.125 | no |  |
| sav / year / full | 1 | 0.875 | 0.750 | -0.125 | no |  |
| refund / accountid / wrong | 3 | 0.875 | 0.833 | -0.042 | no |  |
| growth / pric / plan | 3 | 0.917 | 0.917 | +0.000 | no |  |
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
| invoiceid / paid / client | 3 | 1.000 | 1.000 | +0.000 | no |  |
| payout / tim / bank | 1 | 1.000 | 1.000 | +0.000 | no |  |
| automatically / interest / add | 1 | 1.000 | 1.000 | +0.000 | no |  |
| help / received / chargeback | 1 | 1.000 | 1.000 | +0.000 | no |  |
| fee / card / transaction | 3 | 1.000 | 1.000 | +0.000 | no |  |
| bill / offer / discount | 9 | 0.931 | 0.972 | +0.042 | yes |  |
| month / growth / access | 2 | 0.500 | 0.875 | +0.375 | no |  |
| customer / weekend / open | 1 | 0.000 | 1.000 | +1.000 | no |  |

## Gate: retrieval_outage vs baseline: **FAIL**

- thresholds: overall drop > 0.05, cluster drop > 0.15 (clusters with n >= 5), newly failing items > 5
- combined: 0.9266 -> 0.3732 (delta -0.553, paired bootstrap 95% CI [-0.683, -0.420])
- items newly failing: 29, items fixed: 0
- reason: overall combined dropped by 0.553 (limit 0.05)
- reason: cluster 'bill / offer / discount' (n=9) dropped by 0.403 (limit 0.15)
- reason: 29 items newly failing (limit 5)

| cluster | n | baseline | candidate | delta | gated | regressed |
|---|---|---|---|---|---|---|
| backup / full / workspac | 1 | 1.000 | 0.000 | -1.000 | no |  |
| pay / includ / user | 1 | 1.000 | 0.000 | -1.000 | no |  |
| day / trial / thx | 1 | 1.000 | 0.000 | -1.000 | no |  |
| lin / item / tax | 1 | 1.000 | 0.000 | -1.000 | no |  |
| support / sso / availabl | 1 | 1.000 | 0.000 | -1.000 | no |  |
| sso / plan / whcih | 1 | 1.000 | 0.000 | -1.000 | no |  |
| payment / charg / card | 2 | 1.000 | 0.000 | -1.000 | no |  |
| data / stored / download | 1 | 1.000 | 0.000 | -1.000 | no |  |
| fee / card / transaction | 3 | 1.000 | 0.000 | -1.000 | no |  |
| growth / pric / plan | 3 | 0.917 | 0.000 | -0.917 | no |  |
| lat / fee / maximum | 1 | 1.000 | 0.125 | -0.875 | no |  |
| sav / year / full | 1 | 0.875 | 0.000 | -0.875 | no |  |
| automatically / interest / add | 1 | 1.000 | 0.125 | -0.875 | no |  |
| starter / limit / plan | 1 | 1.000 | 0.250 | -0.750 | no |  |
| email / out / overdu | 1 | 1.000 | 0.292 | -0.708 | no |  |
| payer / up / automatic | 1 | 1.000 | 0.375 | -0.625 | no |  |
| payout / tim / bank | 1 | 1.000 | 0.375 | -0.625 | no |  |
| month / growth / access | 2 | 0.500 | 0.000 | -0.500 | no |  |
| export / invoic / yen | 1 | 1.000 | 0.500 | -0.500 | no |  |
| bill / offer / discount | 9 | 0.931 | 0.528 | -0.403 | yes | YES |
| help / received / chargeback | 1 | 1.000 | 0.875 | -0.125 | no |  |
| refund / accountid / wrong | 3 | 0.875 | 0.833 | -0.042 | no |  |
| trial / fre / card | 2 | 1.000 | 1.000 | +0.000 | no |  |
| customer / weekend / open | 1 | 0.000 | 0.000 | +0.000 | no |  |
| annual / plan / discount | 1 | 1.000 | 1.000 | +0.000 | no |  |
| invoiceid / paid / client | 3 | 1.000 | 1.000 | +0.000 | no |  |
| cup / won / world | 1 | 1.000 | 1.000 | +0.000 | no |  |

## Gate: retrieval_outage vs baseline: **FAIL**

- thresholds: overall drop > 0.05, cluster drop > 0.15 (clusters with n >= 3), newly failing items > 3
- combined: 0.9266 -> 0.3732 (delta -0.553, paired bootstrap 95% CI [-0.683, -0.420])
- items newly failing: 29, items fixed: 0
- reason: overall combined dropped by 0.553 (limit 0.05)
- reason: cluster 'bill / offer / discount' (n=9) dropped by 0.403 (limit 0.15)
- reason: cluster 'growth / pric / plan' (n=3) dropped by 0.917 (limit 0.15)
- reason: cluster 'fee / card / transaction' (n=3) dropped by 1.000 (limit 0.15)
- reason: 29 items newly failing (limit 3)

| cluster | n | baseline | candidate | delta | gated | regressed |
|---|---|---|---|---|---|---|
| backup / full / workspac | 1 | 1.000 | 0.000 | -1.000 | no |  |
| pay / includ / user | 1 | 1.000 | 0.000 | -1.000 | no |  |
| day / trial / thx | 1 | 1.000 | 0.000 | -1.000 | no |  |
| lin / item / tax | 1 | 1.000 | 0.000 | -1.000 | no |  |
| support / sso / availabl | 1 | 1.000 | 0.000 | -1.000 | no |  |
| sso / plan / whcih | 1 | 1.000 | 0.000 | -1.000 | no |  |
| payment / charg / card | 2 | 1.000 | 0.000 | -1.000 | no |  |
| data / stored / download | 1 | 1.000 | 0.000 | -1.000 | no |  |
| fee / card / transaction | 3 | 1.000 | 0.000 | -1.000 | yes | YES |
| growth / pric / plan | 3 | 0.917 | 0.000 | -0.917 | yes | YES |
| lat / fee / maximum | 1 | 1.000 | 0.125 | -0.875 | no |  |
| sav / year / full | 1 | 0.875 | 0.000 | -0.875 | no |  |
| automatically / interest / add | 1 | 1.000 | 0.125 | -0.875 | no |  |
| starter / limit / plan | 1 | 1.000 | 0.250 | -0.750 | no |  |
| email / out / overdu | 1 | 1.000 | 0.292 | -0.708 | no |  |
| payer / up / automatic | 1 | 1.000 | 0.375 | -0.625 | no |  |
| payout / tim / bank | 1 | 1.000 | 0.375 | -0.625 | no |  |
| month / growth / access | 2 | 0.500 | 0.000 | -0.500 | no |  |
| export / invoic / yen | 1 | 1.000 | 0.500 | -0.500 | no |  |
| bill / offer / discount | 9 | 0.931 | 0.528 | -0.403 | yes | YES |
| help / received / chargeback | 1 | 1.000 | 0.875 | -0.125 | no |  |
| refund / accountid / wrong | 3 | 0.875 | 0.833 | -0.042 | yes |  |
| trial / fre / card | 2 | 1.000 | 1.000 | +0.000 | no |  |
| customer / weekend / open | 1 | 0.000 | 0.000 | +0.000 | no |  |
| annual / plan / discount | 1 | 1.000 | 1.000 | +0.000 | no |  |
| invoiceid / paid / client | 3 | 1.000 | 1.000 | +0.000 | yes |  |
| cup / won / world | 1 | 1.000 | 1.000 | +0.000 | no |  |

## Timing

| step | wall time (s) |
|---|---|
| ingest | 0.0 |
| cluster | 0.25 |
| draft | 582.2 |
| review | 0.0 |
| score: baseline predict | 207.8 |
| score: baseline judge | 327.2 |
| score: candidate predict | 209.2 |
| score: candidate judge | 386.0 |
| score: prompt_change predict | 241.9 |
| score: prompt_change judge | 298.9 |
| score: retrieval_outage predict | 203.8 |
| score: retrieval_outage judge | 374.8 |
| run_all | 2832.3 |
