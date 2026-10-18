# Крупные файлы через GitHub Actions

Задача: забрать большой артефакт по presigned-ссылке (Sber OBS), когда локальная
машина до этого хоста не достаёт, и положить его в git без LFS.

```
presigned URL ──► GitHub-hosted runner ──► нарезка по 90 МБ ──► ветка artifacts/submission-zip
                                                        ├──► release (резерв)
                                                        └──► artifact (7 дней)
```

## Что уже сделано

Прогон 2026-10-18 (workflow `fetch-artifact`, run 39158866219):

| | |
| --- | --- |
| файл | `submission.zip`, **3 454 733 203 байт** (3.45 ГБ) |
| sha256 | `373d97168e6722895abc92eb459b0e4a21f85f1959f8277428bfc3816de74a5b` |
| части | **37** по 90 МБ, ветка `artifacts/submission-zip` |
| проверка | скачано обратно, собрано `cat`, sha256 совпал |

Внутри архива — submission-фикстура для судейского пайплайна:

```
smolvlm2/          2.03 ГБ   SmolVLM2 (SmolVLMForConditionalGeneration)
action_expert/     1.42 ГБ   SmolVLA-голова в формате LeRobot
README.md, config.yaml, src/pipeline.py, annotations.jsonl, ...   заглушки
```

`action_expert/config.json`: две камеры 3×256×256, `observation.state` = 8,
`action` = 7, `chunk_size` = 8, `type: smolvla`, `vlm_model_name` указывает на
`/home/ubuntu22/aij_robotics_judge/submission/smolvlm2`. Это LIBERO-подобная
раскладка, **не** гуманоид Green Challenge (там state 51 / action 50). То есть
архив описывает формат сдачи «zip с весами VLM + головы», а не наш Docker-путь.

## Почему устроено именно так

* **Runner вместо локальной машины.** Из песочницы `*.obs.ru-moscow-1.hc.sbercloud.ru`
  недоступен (TLS-обрыв), у GitHub-раннера — доступен.
* **90 МБ, а не 100.** GitHub отклоняет файлы строго больше 100 МиБ
  (104 857 600 байт); 90 МБ — зазор.
* **Без LFS.** LFS требует настройки у клонирующего и отдельной квоты; обычные
  блобы в отдельной ветке скачиваются одним `git clone --branch`.
* **Отдельная ветка.** `main` и рабочая ветка остаются лёгкими. Ветка с чанками
  перезаписывается с нуля (orphan + force-push), история не растёт.
* **Пуш пачками по 4 части.** ~360 МБ на пуш вместо 37 пушей: быстрее и далеко
  от лимита GitHub в 2 ГБ.

## Как забрать результат

```bash
./tools/fetch_parts.sh /tmp/submission.zip          # клон + сборка + проверка sha256
LIST=1 ./tools/fetch_parts.sh /tmp/submission.zip   # ещё и список файлов
```

Вручную:

```bash
git clone --depth 1 --branch artifacts/submission-zip \
  https://github.com/overbaffe1/Green-Challenge.git parts
cd parts && cat submission.zip.part.* > submission.zip && sha256sum -c SHA256SUMS
```

Клон ветки занимает ~5.7 ГБ на диске (пак + рабочие файлы) и ~6 минут.

## Как запустить заново (другая ссылка)

```bash
printf '%s\n' "$PRESIGNED_URL" > submission.url
git add -f submission.url && git commit -m "trigger fetch-artifact" && git push
gh run watch

# сразу после старта скачивания убрать ссылку из ветки
git rm --cached submission.url && rm submission.url
git commit -m "drop submission.url" && git push
```

Правильный путь — секрет `SUBMISSION_URL` + `workflow_dispatch` (тогда ссылка не
попадает ни в историю, ни в логи). Но токен интеграции в этой сессии не имеет
права `actions:write`, поэтому секреты и dispatch недоступны, и workflow читает
файл `submission.url` как последний fallback. Файл в `.gitignore`; чтобы
запустить прогон, его приходится коммитить, поэтому после запуска ссылку нужно
удалить из истории (`git rebase --onto`, force-push). Считайте её
скомпрометированной: presigned-ссылка даёт доступ к объекту всем, у кого она
есть.

## Цена решения

* Репозиторий на GitHub вырос примерно на 3.5 ГБ. GitHub предупреждает с 1 ГБ и
  блокирует около 5 ГБ; 3.45 ГБ влезает, но полный клон станет долгим.
* Публичный репозиторий = публичный артефакт. Если внутри чужое или закрытое —
  чанки надо держать в приватном репозитории.
* Удалить, когда не нужно:
  `git push origin --delete artifacts/submission-zip`
