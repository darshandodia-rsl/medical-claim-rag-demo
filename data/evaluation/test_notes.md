# Test notes for the Meridian Care Plus knowledge base

Keep this file out of the retrieval corpus. It is the answer key for building test claims.

## Deliberate gaps (the policy never mentions these)

| Situation | Nearest clauses | Why it is ambiguous |
|---|---|---|
| Telemedicine or video consultation | 5.1 | Covers "fees of a Medical Practitioner for consultation" with no mention of mode. |
| Acupuncture, chiropractic, reiki, other non-AYUSH therapies | 1.13, 3.6, 10.16 | AYUSH is defined; nothing says whether other alternative therapy is covered or excluded. Only "unregistered practitioners" (10.16) might catch it. |
| Experimental stem-cell therapy | 1.7 | There is no experimental-treatment exclusion. Only the Medical Necessity test ("generally accepted standards") bears on it. |
| Robotic-assisted surgery | 3.1, 1.7 | No mention for or against. |
| Outpatient psychotherapy or counselling | 5.1 | Not named. 10.7 and 3.7 touch psychiatric or substance matters only. |
| Home nursing or hospital-at-home beyond 3.7 | 3.7 | 3.7 is narrow (3+ days, listed exclusions) and does not address new care models. |
| Emergency illness treated for under 24 hours without a Day-Care listing | 1.5, 7.6 | 7.6 covers Accidents only. An illness with a short emergency stay falls between 1.5 and 4.3. |

## Conflicts and precedence (the policy says which wins)

- **Ambulance:** 7.3 (INR 3,000 per Emergency event) against 11.4 (INR 2,000 per Hospitalization). 11.9 says the specific clause 7.3 wins.
- **Specified condition that is also pre-existing:** 8.3 (24 months) against 9.1 (36 months). 8.5 says 36 months applies.
- **Room rent cap and maternity:** 11.1 and 11.3 against 6.2. 6.2 says the room rent cap does not apply to maternity.
- **Accident and waiting periods:** 8.2 waives 8.1 and 8.3 for Accidents but not 8.4 or 9.1. 9.6 then allows Accident treatment on a body part with a pre-existing condition, apportioned.
- **Declared PED after 36 months:** 9.4 says covered, but 9.5 preserves Company rights under 15.1 until the 60-month moratorium in 15.3 ends.
- **Schedule against body:** 2.5.4 says Schedule amounts prevail, and the body governs conditions.

## Exclusions with exceptions

10.2 / 10.2.1 (cosmetic, with accident, burns, cancer reconstruction), 10.3 / 10.3.1 / 10.3.2 (dental), 10.4 (refractive error over 7.5 dioptres, and 4.4 "notwithstanding 10.4" for cataract lens), 10.5 (bariatric, BMI test plus 36 months), 10.7 (poisoning), 10.8 (recreational against hazardous), 10.10 (packaged consumables), 10.11 (diagnostics that lead to treatment), 10.13 with 5.4 (vaccines), 10.14 (circumcision), 10.15 with 3.1(e) (implants).

## Soft spots worth probing

- 10.3.1 allows outpatient dental after an accident "within the limit in clause 7.6", but 7.6 itself requires treatment within 24 hours of the Accident, while 10.3.1 allows 90 days.
- 8.3(m) puts Congenital Internal Anomalies on a 24-month wait while 10.12 excludes only External ones.
- 4.3 unlisted day-care procedures need prior written confirmation, so a procedure done without it is a judgment call.
- 6.4 against 6.2: whether a complication is "before labour" or "from labour onwards" decides which limit applies.

## Worked example for the calculation order (14.6)

Plan A (Sum Insured INR 5,00,000), insured aged 65, Network Hospital, 3-day stay in a room costing INR 8,000 per day. Room limit is INR 5,000 per day (11.1).

| Item | Billed | Payable after Step 2 |
|---|---|---|
| Room rent (3 days) | 24,000 | 15,000 |
| Surgeon (room-linked, ratio 5,000/8,000 = 0.625) | 60,000 | 37,500 |
| Operation theatre (room-linked) | 20,000 | 12,500 |
| Medicines (not room-linked) | 30,000 | 30,000 |
| **Total** | 1,34,000 | **95,000** |

No deductible. 20% age co-payment (11.6) leaves 76,000 payable.

## Labels and payable amounts

`expected_decision` takes `approve`, `partial`, `reject` or `needs_review`. `expected_payable` is a number for the first three and `null` for `needs_review`.

- `approve`: admissible and payable in full (payable equals amount claimed).
- `partial`: admissible but reduced by a sub-limit, co-pay or deduction (C004, C007, C013, C019, C023).
- `reject`: payable is 0.
- `needs_review`: the policy is silent or ambiguous, so no payable is asserted.

Calculation basis for the partial claims, following clause 14.6:

| Claim | Calculation | Payable |
|---|---|---|
| C004 | Room 4 x 5,000 = 20,000; doctor and nursing 18,000 x 5,000/6,500 = 13,846; tests and medicines 34,000 | 67,846 |
| C007 | 165,000 less 30% cumulative co-pay (11.8) | 115,500 |
| C013 | Per-eye cap on Plan B (standard-lens price 38,000 exceeds it) | 30,000 |
| C019 | 88,000 less 10% for missing pre-authorization (13.4.1) | 79,200 |
| C023 | Normal-delivery limit on Plan B (6.2.1) | 60,000 |
