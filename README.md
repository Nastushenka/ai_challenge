# LLM Chat — веб-интерфейс DeepSeek

Сторонние библиотеки не нужны.

```bash
cd web-simple
cp .env.example .env
# Впишите ключ в LLM_API_KEY внутри .env
python3 server.py
```

Откройте в браузере: <http://127.0.0.1:8000>

По умолчанию используется `deepseek-v4-pro` через официальный Responses API
DeepSeek. Адрес API и модель настраиваются в локальном файле `.env`:

- `LLM_API_KEY` — API-ключ;
- `LLM_BASE_URL` — `https://api.deepseek.com`;
- `LLM_MODEL` — `deepseek-v4-pro`.

Файл `.env` исключён из Git и не попадёт в репозиторий.
