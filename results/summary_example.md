# Fleet reliability forecast

Ingest accepted 711,935 inspection records and quarantined 7,073 (0.98%) with reason codes; nothing was silently repaired.
Sensor screening flagged 341 channels (bias_step 109, scale_error 7, stuck 96, dropout 91, accelerated 38) and placed 550 more on a watch list with weak, unconfirmed evidence of a rate change. Only the classes that corrupt the reading itself (bias step, scale error, stuck) are excluded from the fleet fit and have their damage state estimated from usage; dropout channels keep the readings that did arrive, and accelerated channels are kept in full, because accelerated wear is real damage rather than a sensor fault.

Forecasts were issued for 12,000 installed components over a 18-month horizon. The expected number of failures in that window is 5773.2; 5573 components exceed the 0.5 failure-probability threshold and are listed in the ledger with their evidence.
Monte Carlo fleet availability is 75.1% in month 1 and 51.0% in month 18 (90% band 48.9% to 52.9%).

## Top risk

- T0000 23A: P(fail) 1.0, damage basis reading, sensor none
- T0001 64E: P(fail) 1.0, damage basis reading, sensor none
- T0001 23A: P(fail) 1.0, damage basis reading, sensor none
- T0002 75G: P(fail) 1.0, damage basis reading, sensor none
- T0003 64E: P(fail) 1.0, damage basis reading, sensor none
- T0005 23A: P(fail) 1.0, damage basis reading, sensor none
- T0007 64E: P(fail) 1.0, damage basis reading, sensor none
- T0008 64E: P(fail) 1.0, damage basis reading, sensor none
- T0008 75G: P(fail) 1.0, damage basis reading, sensor none
- T0010 64E: P(fail) 1.0, damage basis reading, sensor none
