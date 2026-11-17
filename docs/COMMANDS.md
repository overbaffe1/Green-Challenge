# Команды с нуля

Пошагово, на своей машине с GPU. Копируйте блоки целиком. Места, где нужно
подставить своё, помечены `‹…›`.

Ориентиры по ресурсам: **≥ 150 ГБ свободного диска** (комфортно 200+), GPU с
≥ 24 ГБ VRAM, Ubuntu 22.04/24.04, ~1 ч на первый build образа симулятора и ещё
~20–40 мин на скачивание ассетов.

### Откуда берутся 150 ГБ

Это не требование из документации, а сумма того, что реально оседает на диске.
Жёсткий минимум ниже, запас нужен, чтобы Docker не упал на середине слоя.

| Что | Типичный размер | Как уменьшить |
| --- | --- | --- |
| `nvcr.io/nvidia/isaac-lab:2.3.2` (база симулятора) | 30–50 ГБ распакованного | ничего, это обязательная база |
| образ `green-challenge:latest` (база + Isaac Lab + ассеты внутри) | ещё 10–30 ГБ сверху | ассеты монтировать с диска вместо `COPY` |
| образ политики `green-challenge:1.3` | до 50 ГБ по правилам конкурса | он один, второй не нужен |
| датасет `GreenChallengeData` | неизвестен, десятки ГБ | проверить размер до скачивания (ниже) |
| кэш HF, если качали без `--local-dir` | до 2× датасета | `export HF_HUB_CACHE=...` на большой диск |
| build cache Docker | 0.3–1× суммы образов | `docker builder prune` после сборки |
| временные файлы Isaac (OV-кэш, шейдеры) | 5–15 ГБ | чистится вместе с контейнером |

Проверить размер датасета **до** скачивания (≈10 секунд):

```bash
curl -s "https://huggingface.co/api/datasets/SberRoboticsCenter/GreenChallengeData/tree/main?recursive=true" \
| python3 -c 'import json,sys; rows=json.load(sys.stdin); print("%.1f GB, %d files" % (sum(r.get("size",0) for r in rows)/1e9, len(rows)))'
```

То же для ассетов — замените имя на `GreenChallengeAssets`.

Измерить, что уже занято на машине:

```bash
docker system df                      # образы, контейнеры, build cache
docker image ls --format '{{.Repository}}:{{.Tag}}\t{{.Size}}'
du -sh ~/.cache/huggingface 2>/dev/null
du -sh ~/green_challenge/data ~/green_challenge/assets 2>/dev/null
```

Практическое правило: если свободно **меньше 90 ГБ**, всё ещё можно — но тогда
обязательно `docker builder prune` после сборки, ассеты на внешнем диске и
`HF_HUB_CACHE` туда же. Если меньше 60 ГБ — сборка образа симулятора не пройдёт.

---

## 0. Проверка машины

```bash
git clone https://github.com/overbaffe1/Green-Challenge.git
cd Green-Challenge
./tools/check_host.sh
```

Скрипт проверяет драйвер, `docker`, `docker compose`, GPU-рантайм, диск и RAM и
печатает, что именно чинить. Если `docker`/`nvidia-smi` нет — сначала установите:

```bash
# драйвер + контейнерный тулкит (Ubuntu)
sudo apt-get update
sudo apt-get install -y nvidia-driver-570 nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo usermod -aG docker "$USER"      # затем перелогиньтесь
```

Проверка, что контейнеры видят GPU:

```bash
docker run --rm --gpus all ubuntu:22.04 nvidia-smi
```

---

## 1. Тесты (без GPU, 10 секунд)

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
pytest tests -q
python tools/smoke_test.py --seconds 1.6 --speed 1.3
```

Ожидаемо: `56 passed`, затем `live plans: 15 peak`. Это проверяет транспорт,
декодирование эха и фид. Если тут падает — дальше идти смысла нет.

---

## 2. Базовый образ конкурса

```bash
docker login aij-green-challenge.cr.cloud.ru     # если реестр просит логин
docker pull aij-green-challenge.cr.cloud.ru/green-vla:latest

# запомните digest — его и надо подставлять в сборку и в сабмит
docker image inspect --format '{{index .RepoDigests 0}}' \
  aij-green-challenge.cr.cloud.ru/green-vla:latest
```

Вывод вида `aij-green-challenge.cr.cloud.ru/green-vla@sha256:317dc67f…`.
В задании указан именно digest; тег `:latest` может измениться.

---

## 3. Сборка нашего образа

```bash
export BASE='aij-green-challenge.cr.cloud.ru/green-vla@sha256:‹ПОЛНЫЙ_DIGEST›'
docker build -f docker/Dockerfile --build-arg BASE_IMAGE="$BASE" -t green-challenge:1.3 .
```

Если digest из задания недоступен, соберите от `:latest` — семантика та же, но
для сабмита лучше digest:

```bash
make docker-build          # использует тег по умолчанию
```

Размер (должен быть < 50 ГБ):

```bash
docker image inspect green-challenge:1.3 --format '{{.Size}}' | awk '{printf "%.1f GB\n", $1/1e9}'
```

---

## 4. Проверка образа без симулятора

```bash
docker run --gpus all --rm --network none -p 8999:8999 \
  -e POLICY_PORT=8999 -e POLICY_RAW_PORT=8998 -e GC_SPEED=1.3 \
  green-challenge:1.3
```

В другом терминале:

```bash
curl -sf http://127.0.0.1:8999/healthz && echo " OK"
```

В логе должны быть строки `configuration: {... 'stack': 'official', 'speed': 1.3 ...}`,
`starting raw policy: ... --feed-by-server --feed.row-dt 0.025641...`, `ready: ws://0.0.0.0:8999`.
`--network none` доказывает, что образ ничего не докачивает при старте.

Остановить: `Ctrl+C` в первом терминале.

---

## 5. Симулятор и ассеты

```bash
cd ~
git clone https://github.com/greenvla/GreenSim.git green_challenge
cd green_challenge

# ~30–60 мин: Isaac Sim / Isaac Lab
make build

# ассеты (USD-модели) и датасет
pip install -U "huggingface_hub[cli]"
hf download SberRoboticsCenter/GreenChallengeAssets --repo-type dataset --local-dir assets/usd_models
hf download SberRoboticsCenter/GreenChallengeData   --repo-type dataset --local-dir data
```

Со старым CLI те же команды называются `huggingface-cli download ...`.
Если репозитории закрытые: `hf auth login` с токеном.

Проверьте, что структура на месте:

```bash
ls assets/usd_models | head
ls data/meta/info.json
```

---

## 6. Запуск: baseline 1.0× и наш 1.3×

Политика и симулятор должны быть на одном хосте: контейнер симулятора работает с
`network_mode: host` и стучится в `127.0.0.1:8999`. Поэтому политику тоже
запускаем с `--network host`.

**Терминал A — политика, baseline:**

```bash
docker run --gpus all --rm --network host \
  -e POLICY_PORT=8999 -e POLICY_RAW_PORT=8998 -e GC_SPEED=1.0 \
  -e GC_TELEMETRY=/tmp/greenchallenge/baseline.jsonl \
  green-challenge:1.3
```

**Терминал B — бенчмарк (8 эпизодов × 3 сценария, до 4 ч):**

```bash
cd ~/green_challenge
POLICY_PORT=8999 \
RUN_PATH=assets/runs/green_challenge.json \
RESULTS_PATH=/workspace/green_challenge/results.baseline.json \
make up
```

`make up` сам поднимает контейнер симулятора и запускает бенчмарк.
`RESULTS_PATH` должен быть внутри смонтированного репозитория (контейнер видит
его как `/workspace/green_challenge/...`), поэтому в примере такой путь; на хосте
файл появится как `~/green_challenge/results.baseline.json`.

То же самое через нашу обёртку (она добавит метку времени, если путь не задан):

```bash
GREEN_SIM_DIR=~/green_challenge POLICY_PORT=8999 \
RESULTS_PATH=~/green_challenge/results.baseline.json \
~/Green-Challenge/eval/run_benchmark.sh
```

Повторный `make up` перед вторым прогоном не нужен: контейнер уже поднят, а
`make up` идемпотентен и просто запустит бенчмарк заново.

Посмотреть, что получилось:

```bash
python ~/Green-Challenge/eval/report.py ~/green_challenge/results.baseline.json --sweep
```

**Терминал A — остановить baseline (`Ctrl+C`), запустить 1.3×:**

```bash
docker run --gpus all --rm --network host \
  -e POLICY_PORT=8999 -e POLICY_RAW_PORT=8998 -e GC_SPEED=1.3 \
  -e GC_TELEMETRY=/tmp/greenchallenge/fast.jsonl \
  green-challenge:1.3
```

**Терминал B — тот же бенчмарк ещё раз:**

```bash
cd ~/green_challenge
POLICY_PORT=8999 RUN_BENCHMARK=1 \
RUN_PATH=assets/runs/green_challenge.json \
RESULTS_PATH=/workspace/green_challenge/results.fast130.json \
make up
```

---

## 7. Сравнение и ответ на вопрос «выше 68.14?»

```bash
python ~/Green-Challenge/eval/report.py \
  ~/green_challenge/results.baseline.json \
  ~/green_challenge/results.fast130.json \
  --compare --sweep
```

Что смотреть, по порядку:

1. `score` во второй строке каждого отчёта — это и есть конкурсный скор.
2. `subtasks: N/88 completed` — завершаемость. Если она упала при 1.3×, ускорение
   надо откатывать, а не наращивать.
3. Дельту по сценариям в `--compare`.

По этим двум прогонам сразу видно, попадаем ли мы в цель:

| Что показал прогон | Что делать |
| --- | --- |
| 1.3× ≥ 68.14 и завершаемость не хуже | Сабмит слота 1: `GC_SPEED=1.3`, образ по digest. |
| 1.3× < 68.14, но завершаемость 100% | Не хватает скорости: пробовать 1.4–1.5 и `GC_EXPIRE_FROZEN=1`; бюджет по `docs/SCORE.md`. |
| завершаемость < 90% | Скорость не спасёт: нужен файнтюн (`docs/TRAINING.md`), иначе цель недостижима. |
| 1.0× уже близко к 68 | Проверить, что симулятор запущен с `PROPRIO_CMD=all`; иначе числа не сравнить с публичными. |

---

## 8. Если что-то падает

| Симптом | Что проверить |
| --- | --- |
| `docker: Error response from daemon: could not select device driver` | не установлен `nvidia-container-toolkit`; `nvidia-ctk runtime configure --runtime=docker && systemctl restart docker` |
| Образ падает на старте | в логах трейсбек дочернего процесса; чаще всего нет `/ckpt` или не совпал драйвер/CUDA |
| `/healthz` молчит дольше 10 мин | холодная загрузка чекпоинта; поднимите `GC_START_TIMEOUT=1200` |
| Симулятор подключается, ответов нет | политика не с `--network host`, либо порт не 8999 |
| `make up` не видит GPU | `docker compose` v2 и `deploy.resources` требуют свежий compose; проверьте `docker compose version` |
| Скор ниже, чем при 1.0× | сработал блендинг на устаревших планах: `GC_BLEND_LIFE=1.0` или `GC_EXPIRE_FROZEN=1` |
| Нужен «чистый» baseline | `--entrypoint python green-challenge:1.3 /app/docker/entrypoint.official.py` (в образе сохранён оригинальный entrypoint) |

---

## 9. Отправка сабмита

```bash
# тег с датой, чтобы не перепутать сабмиты
docker tag green-challenge:1.3 ‹REGISTRY›/green-challenge:2026-10-18-130
docker push ‹REGISTRY›/green-challenge:2026-10-18-130
docker image inspect --format '{{index .RepoDigests 0}}' \
  ‹REGISTRY›/green-challenge:2026-10-18-130
```

В файл сабмита вставляется **digest-ссылка**, а не тег: тег можно перезаписать,
digest — нет. Один сабмит в сутки, лучший результат считается; план трёх слотов —
в `docs/RUNBOOK.md`.
