### Before the solution
<img width="1218" height="363" alt="image" src="https://github.com/user-attachments/assets/6ed743db-af53-484a-8529-9ac4b30c24b2" />


### After 

### Result of question1/run.py
```
{
  "resolutions": [
    {
      "ticket_id": "T-1001",
      "action": "REFUND",
      "kb_article": "returns-within-30-days"
    },
    {
      "ticket_id": "T-1002",
      "action": "SEND_ARTICLE",
      "kb_article": "digital-license-no-refund"
    },
    {
      "ticket_id": "T-1003",
      "action": "ESCALATE_TIER2",
      "kb_article": "custom-items-no-replace"
    },
    {
      "ticket_id": "T-1004",
      "action": "REPLACE",
      "kb_article": "damage-replacement-7-days"
    }
  ],
  "total_points": 6216
}
```
Means 1554 points per ticket which is <3000.

### Result of question2/run.py
```
{
  "resolutions": [
    {
      "ticket_id": "T-1001",
      "action": "REFUND",
      "kb_article": "returns-within-30-days"
    },
    {
      "ticket_id": "T-1002",
      "action": "SEND_ARTICLE",
      "kb_article": "digital-license-no-refund"
    },
    {
      "ticket_id": "T-1003",
      "action": "ESCALATE_TIER2",
      "kb_article": "custom-items-no-replace"
    },
    {
      "ticket_id": "T-1004",
      "action": "REPLACE",
      "kb_article": "damage-replacement-7-days"
    }
  ],
  "total_points": 5490
}
```
Means 1373 points per ticket which is <2500.

### Final Observation 
```
# Q1 T-1001

call            in_tok  cached   out_tok   in_pts  out_pts    total
handle_ticket      330       0      46        990      690     1680

# Q2 T-1002

call            in_tok  cached   out_tok   in_pts  out_pts    total
compose            262       0        39      786      585     1371
```


* test_baseline_cost_is_high (both questions) asserts the baseline is expensive, so it fails by design once the agent is optimised. I added tests/test_budget.py with the real targets instead.


 
