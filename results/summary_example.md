# Fleet reliability forecast

Ingest accepted 719,008 inspection records and quarantined 7,073 (0.98%) with reason codes; nothing was silently repaired.
Sensor screening flagged 338 channels (bias_step 117, scale_error 3, stuck 96, dropout 91, accelerated 31) and placed 416 more on a watch list with weak, unconfirmed evidence of a rate change. Flagged channels were excluded from the fleet fit and their damage state was estimated from usage rather than from the reading.

Forecasts were issued for 12,000 installed components over a 18-month horizon. The expected number of failures in that window is 5963.5; 5761 components exceed the 0.5 failure-probability threshold and are listed in the ledger with their evidence.
Monte Carlo fleet availability is 74.5% in month 1 and 49.3% in month 18 (90% band 47.1% to 51.2%).

## Top risk

- T0000 23A: P(fail) 1.0, damage basis reading, sensor none
- T0001 64E: P(fail) 1.0, damage basis reading, sensor none
- T0001 23A: P(fail) 1.0, damage basis reading, sensor none
- T0002 75G: P(fail) 1.0, damage basis reading, sensor none
- T0003 64E: P(fail) 1.0, damage basis reading, sensor none
- T0007 64E: P(fail) 1.0, damage basis reading, sensor none
- T0008 64E: P(fail) 1.0, damage basis reading, sensor none
- T0008 75G: P(fail) 1.0, damage basis reading, sensor none
- T0010 52C: P(fail) 1.0, damage basis reading, sensor none
- T0010 75G: P(fail) 1.0, damage basis reading, sensor none
