# makefun-gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nuannan13579/makefun-gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nuannan13579/makefun-gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A launch-day curve forecaster for Twitch viewership. Instead of assuming a game holds its first-hour audience, the program learns the shape of a release day from the case's own past-launch comparables: for each comparable it records, hour by hour, the log-ratio of that hour's viewers to the launch's hour-1 viewers. Because launches that opened at a similar scale behave more alike, each comparable is weighted by a Gaussian kernel on the distance between its hour-1 size and the case's hour-1 print in log space, and the five forecast quantiles for each hour come from a weighted normal fit to that hour's log-ratio distribution, anchored on the case's actual first-hour count. Hours with too few comparable samples pool their neighbours, and a widening-spread fallback covers degenerate inputs, so the program always emits a valid forecast. The constants were chosen by leave-one-out validation over the pooled practice comparables (tools/loo.mjs), where this model scored 0.392 mean pinball loss on the log scale against 0.930 for a persistence baseline on the same folds. It runs offline on Node 22 with the standard library only and is fully deterministic.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---
<sub>Entered on [make.fun](https://make.fun/?ref=9959qcny&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
